import os
import os.path as path
import re
import html
import json
import time
import argparse
import mask2contour
import numpy as np
import torch.utils as utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visualizer import Visualizer
from ice_dataset import ICEDataset
from pix2pix_model import Pix2PixModel

parser = argparse.ArgumentParser(
    description="Test a face inpainting model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--name", default="2d_no_clips",
    help="the name of the experiment"
)
parser.add_argument(
    "--checkpoints_dir", default="checkpoints",
    help="a folder that contains all the trained model checkpoints"
)
parser.add_argument(
    "--data_dir", default="data/test",
    help="a folder that contains all the prepared testing ICE data"
)
parser.add_argument(
    "--image_dir", default="data/usd_images",
    help="a folder that contains all prepocessed ICE data"
)
parser.add_argument(
    "--results_dir", default="results/",
    help="a folder that stores all the experimental results"
)
parser.add_argument(
    "--dice_only", default=False, action="store_true",
    help="if specified, compute the dice score only"
)
parser.add_argument(
    "--batch_size", type=int, default=16,
    help="number of images per mini-batch"
)
parser.add_argument(
    "--num_workers", type=int, default=8,
    help="number of workers to load images"
)
parser.add_argument(
    "--image_size", type=int, default=256,
    help="the size of the input volumes"
)
parser.add_argument(
    "--map_nc", type=int, default=6,
    help="number of landmark channels"
)
parser.add_argument(
    "--ngf", type=int, default=128,
    help="number of basic generator features"
)
parser.add_argument(
    "--ndf", type=int, default=64,
    help="number of basic discriminator features"
)
parser.add_argument(
    "--norm_layer", default="batch",
    help="the type of the normalization layer"
)
parser.add_argument(
    "--unet_size", type=int, default=8,
    help="number of downsampling/upsampling layers in unet"
)
parser.add_argument(
    "--phase", default="test",
    help="the phase of the experiment"
)
parser.add_argument(
    "--which_epoch", default="24",
    help="the epoch number used to resume the training"
)
parser.add_argument(
    "--gpu_ids", type=int, nargs="*", default=[0],
    help="the gpu devices used for training/testing"
)
args = parser.parse_args()


def project_contour2world(contour, image2world_matrix):
    contour = np.concatenate(
        (
            contour,
            np.zeros((contour.shape[0], 1)),
            np.ones((contour.shape[0], 1))
        ), axis=1
    )
    contour = np.dot(
        contour, image2world_matrix.T
    )[:, :3]
    return contour


def compute_contour_error(contour_A, contour_B):
    A2B_distances = []
    for point_A in contour_A:
        A2B_distances.append(np.sqrt(
            ((contour_B - point_A) ** 2).sum(1)
        ).min())
    avg_A2B_distance = np.mean(A2B_distances)

    B2A_distances = []
    for point_B in contour_B:
        B2A_distances.append(np.sqrt(
            ((contour_A - point_B) ** 2).sum(1)
        ).min())
    avg_B2A_distance = np.mean(B2A_distances)
    return (avg_A2B_distance + avg_B2A_distance) / 2.0


colors = np.array([
    (31, 119, 180),  # blue LA
    (44, 160, 44),  # green LAA
    (214, 39, 40),  # red LIPV
    (255, 127, 14),  # orange LSPV
    (148, 103, 189),  # purple RIPV
    (140, 86, 75),  # brown RSPV
    (0, 0, 0)  # black
]) / 255.0
organ_id2name = {
    0: "LA", 1: "LAA", 2: "LIPV", 3: "LSPV", 4: "RIPV", 5: "RSPV"
}

checkpoints_dir = os.path.join(args.checkpoints_dir, args.name)

exp_dir = path.join(args.results_dir, args.name + "_" + args.which_epoch)
if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)

dataset = ICEDataset(args.data_dir)
dataset_size = len(dataset)
model = Pix2PixModel(
    name=args.name, phase=args.phase, which_epoch=args.which_epoch,
    batch_size=args.batch_size, image_size=args.image_size,
    input_nc=3 + args.map_nc, map_nc=args.map_nc, num_downs=args.unet_size,
    ngf=args.ngf, ndf=args.ndf, gpu_ids=args.gpu_ids,
    checkpoints_dir=args.checkpoints_dir
)

data_loader = utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True
)

# test
index = 0
total_binary_dice = {
    "projected": 0, "reconstructed": 0, "predicted": 0
}
total_class_dice = {
    "projected": np.zeros(6),
    "reconstructed": np.zeros(6),
    "predicted": np.zeros(6)
}
total_contour_error = {
    "projected": 0, "reconstructed": 0, "predicted": 0
}
total_class_error = {
    "projected": np.zeros(6),
    "reconstructed": np.zeros(6),
    "predicted": np.zeros(6)
}
count = 0 + np.finfo(float).eps
class_count = np.zeros(6) + np.finfo(float).eps
error_count = 0
error_class_count = np.zeros(6) + np.finfo(float).eps

for i, data in enumerate(data_loader):
    model.set_input(
        data['image'], data['segmentation_outlined'], data['image_name']
    )
    model.test()

    images = data['image'].numpy()
    images = images * 0.5 + 0.5
    images = images.transpose(0, 2, 3, 1)
    image_names = data['image_name']

    organs = data['organs'].numpy()
    hardness = data['hardness'].numpy()

    segmentation_outlined = data['segmentation_outlined'].numpy()
    segmentation_outlined = segmentation_outlined * 0.5 + 0.5

    segmentation_projected = data['segmentation_projected'].numpy()
    segmentation_projected = segmentation_projected * 0.5 + 0.5

    segmentation_reconstructed = data['segmentation_reconstructed'].numpy()
    segmentation_reconstructed = segmentation_reconstructed * 0.5 + 0.5

    segmentation_predicted = model.fake_map.data.cpu().numpy()
    segmentation_predicted = segmentation_predicted * 0.5 + 0.5
    segmentation_predicted = (segmentation_predicted > 0.5).astype(np.float32)

    for image_index, image_name in enumerate(image_names):
        if args.dice_only:
            break
        index += 1
        print("[{}/{}] Testing {}".format(
            index, dataset_size, image_name
        ))
        image_name = path.split(image_name)[-1]
        subject_name = "_".join(image_name.split("_")[:2])
        frame_name = "_".join(image_name.split("_")[2:])
        subject_dir = path.join(exp_dir, subject_name)
        if not path.isdir(subject_dir):
            os.makedirs(subject_dir)

        image2world_file = path.join(
            args.image_dir, subject_name,
            "{}_transformation.npy".format(frame_name)
        )
        image2world_matrix = np.load(image2world_file)

        image = images[image_index][..., :3]
        contours_outlined = [[]] * organs.shape[1]
        contours_projected = [[]] * organs.shape[1]
        contours_predicted = [[]] * organs.shape[1]
        contours_reconstructed = [[]] * organs.shape[1]

        for organ_index, organ in enumerate(organs[image_index]):
            if organ != 0:
                should_compute_error = True

                contour_outlined = mask2contour.convert_mask2contour(
                    segmentation_outlined[image_index][organ_index]
                )
                contour_outlined = mask2contour.get_sparse_contour(
                    contour_outlined
                )
                if len(contour_outlined) > 0:
                    contour_outlined = mask2contour.catmull_rom_chain(
                        contour_outlined
                    )
                    contours_outlined[organ_index] = contour_outlined
                contour_outlined_world = project_contour2world(
                    contour_outlined, image2world_matrix
                )

                contour_projected = mask2contour.convert_mask2contour(
                    segmentation_projected[image_index][organ_index]
                )
                contour_projected = mask2contour.get_sparse_contour(
                    contour_projected
                )
                if len(contour_projected) > 0:
                    contour_projected = mask2contour.catmull_rom_chain(
                        contour_projected
                    )
                    contours_projected[organ_index] = contour_projected
                    contour_projected_world = project_contour2world(
                        contour_projected, image2world_matrix
                    )
                    projected_contour_error = compute_contour_error(
                        contour_projected_world, contour_outlined_world
                    )
                else:
                    should_compute_error = False

                contour_reconstructed = mask2contour.convert_mask2contour(
                    segmentation_reconstructed[image_index][organ_index]
                )
                contour_reconstructed = mask2contour.get_sparse_contour(
                    contour_reconstructed
                )
                if len(contour_reconstructed) > 0:
                    contour_reconstructed = mask2contour.catmull_rom_chain(
                        contour_reconstructed
                    )
                    contours_reconstructed[organ_index] = contour_reconstructed
                    contour_reconstructed_world = project_contour2world(
                        contour_reconstructed, image2world_matrix
                    )
                    reconstructed_contour_error = compute_contour_error(
                        contour_reconstructed_world, contour_outlined_world
                    )
                else:
                    should_compute_error = False

                contour_predicted = mask2contour.convert_mask2contour(
                    segmentation_predicted[image_index][organ_index]
                )
                contour_predicted = mask2contour.get_sparse_contour(
                    contour_predicted
                )
                if len(contour_predicted) > 0:
                    contour_predicted = mask2contour.catmull_rom_chain(
                        contour_predicted
                    )
                    contours_predicted[organ_index] = contour_predicted
                    contour_predicted_world = project_contour2world(
                        contour_predicted, image2world_matrix
                    )
                    predicted_contour_error = compute_contour_error(
                        contour_predicted_world, contour_outlined_world
                    )
                else:
                    should_compute_error = False

                if should_compute_error:
                    total_class_error["projected"][organ_index] += \
                        projected_contour_error
                    total_contour_error["projected"] += \
                        projected_contour_error

                    total_class_error["reconstructed"][organ_index] += \
                        reconstructed_contour_error
                    total_contour_error["reconstructed"] += \
                        reconstructed_contour_error

                    total_class_error["predicted"][organ_index] += \
                        predicted_contour_error
                    total_contour_error["predicted"] += \
                        predicted_contour_error

                    error_count += 1
                    error_class_count[organ_index] += 1

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.set_size_inches(5, 5)
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(image, aspect='auto')
        for contour_index, contour in enumerate(contours_outlined):
            if len(contour) != 0:
                # close the contour
                contour = contour.tolist()
                contour += [contour[0]]
                contour = np.array(contour)

                if hardness[image_index][contour_index] == 1.0:
                    plt.plot(
                        contour[:, 1], contour[:, 0], "--",
                        color=colors[contour_index]
                    )
                else:
                    plt.plot(
                        contour[:, 1], contour[:, 0],
                        color=colors[contour_index]
                    )
        fig.savefig(
            path.join(
                subject_dir, "{}_outlined_contours.png".format(image_name)
            )
        )

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.set_size_inches(5, 5)
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(image, aspect='auto')
        for contour_index, contour in enumerate(contours_projected):
            if len(contour) != 0:
                # close the contour
                contour = contour.tolist()
                contour += [contour[0]]
                contour = np.array(contour)

                if hardness[image_index][contour_index] == 1.0:
                    plt.plot(
                        contour[:, 1], contour[:, 0], "--",
                        color=colors[contour_index]
                    )
                else:
                    plt.plot(
                        contour[:, 1], contour[:, 0],
                        color=colors[contour_index]
                    )
        fig.savefig(
            path.join(
                subject_dir, "{}_projected_contours.png".format(image_name)
            )
        )

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.set_size_inches(5, 5)
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(image, aspect='auto')
        for contour_index, contour in enumerate(contours_reconstructed):
            if len(contour) != 0:
                # close the contour
                contour = contour.tolist()
                contour += [contour[0]]
                contour = np.array(contour)

                if hardness[image_index][contour_index] == 1.0:
                    plt.plot(
                        contour[:, 1], contour[:, 0], "--",
                        color=colors[contour_index]
                    )
                else:
                    plt.plot(
                        contour[:, 1], contour[:, 0],
                        color=colors[contour_index]
                    )
        fig.savefig(
            path.join(
                subject_dir, "{}_reconstructed_contours.png".format(image_name)
            )
        )

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.set_size_inches(5, 5)
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(image, aspect='auto')
        for contour_index, contour in enumerate(contours_predicted):
            if len(contour) != 0:
                # close the contour
                contour = contour.tolist()
                contour += [contour[0]]
                contour = np.array(contour)

                if hardness[image_index][contour_index] == 1.0:
                    plt.plot(
                        contour[:, 1], contour[:, 0], "--",
                        color=colors[contour_index]
                    )
                else:
                    plt.plot(
                        contour[:, 1], contour[:, 0],
                        color=colors[contour_index]
                    )
        fig.savefig(
            path.join(
                subject_dir, "{}_predicted_contours.png".format(image_name)
            )
        )
        plt.close('all')

        # save predicted contour lines
        for organ_index, contour in enumerate(contours_predicted):
            if len(contour) != 0:
                organ_name = organ_id2name[organ_index]
                contour_file = path.join(
                    subject_dir, "{}_{}.json".format(image_name, organ_name)
                )
                with open(contour_file, "w") as f:
                    json.dump(contour.tolist(), f)

    # compute dice coefficient
    organs = organs[..., np.newaxis, np.newaxis]
    segmentation_outlined = segmentation_outlined * organs
    segmentation_projected = segmentation_projected * organs
    segmentation_reconstructed = segmentation_reconstructed * organs
    segmentation_predicted = segmentation_predicted * organs

    class_dice = segmentation_outlined * segmentation_projected
    im_sum = (
        segmentation_outlined.sum(axis=(0, 2, 3)) +
        segmentation_projected.sum(axis=(0, 2, 3))
    )
    class_dice = (class_dice.sum(axis=(0, 2, 3))) * 2 / (
        im_sum + np.finfo(float).eps
    )
    class_dice[im_sum == 0.0] = 1.0

    total_class_dice['projected'] += class_dice * organs.sum(0).squeeze()

    binary_dice = segmentation_outlined * segmentation_projected
    im_sum = (
        segmentation_outlined.sum() + segmentation_projected.sum()
    )
    if im_sum == 0:
        binary_dice = 1.0
    else:
        binary_dice = binary_dice.sum() * 2 / im_sum
    total_binary_dice['projected'] += binary_dice * \
        segmentation_outlined.shape[0]

    class_dice = segmentation_outlined * segmentation_reconstructed
    im_sum = (
        segmentation_outlined.sum(axis=(0, 2, 3)) +
        segmentation_reconstructed.sum(axis=(0, 2, 3))
    )
    class_dice = (class_dice.sum(axis=(0, 2, 3))) * 2 / (
        im_sum + np.finfo(float).eps
    )
    class_dice[im_sum == 0.0] = 1.0
    total_class_dice['reconstructed'] += class_dice * organs.sum(0).squeeze()

    binary_dice = segmentation_outlined * segmentation_reconstructed
    im_sum = (
        segmentation_outlined.sum() + segmentation_reconstructed.sum()
    )
    if im_sum == 0:
        binary_dice = 1.0
    else:
        binary_dice = binary_dice.sum() * 2 / im_sum
    total_binary_dice['reconstructed'] += binary_dice * \
        segmentation_outlined.shape[0]

    class_dice = segmentation_outlined * segmentation_predicted
    im_sum = (
        segmentation_outlined.sum(axis=(0, 2, 3)) +
        segmentation_predicted.sum(axis=(0, 2, 3))
    )
    class_dice = (class_dice.sum(axis=(0, 2, 3))) * 2 / (
        im_sum + np.finfo(float).eps
    )
    class_dice[im_sum == 0.0] = 1.0
    total_class_dice['predicted'] += class_dice * organs.sum(0).squeeze()

    binary_dice = segmentation_outlined * segmentation_predicted
    im_sum = (
        segmentation_outlined.sum() +
        segmentation_predicted.sum()
    )
    if im_sum == 0:
        binary_dice = 1.0
    else:
        binary_dice = binary_dice.sum() * 2 / im_sum
    total_binary_dice['predicted'] += binary_dice * \
        segmentation_outlined.shape[0]

    count += segmentation_outlined.shape[0]
    class_count += organs.sum(0).squeeze()

    print(
        (
            "[{}/{}] Binary dice (projected) : {}, "
            "class dice (projected) : {}"
        ).format(
            i, len(dataset) / args.batch_size,
            total_binary_dice['projected'] / count,
            total_class_dice['projected'] / class_count
        )
    )

    print(
        (
            "[{}/{}] Total error (projected) : {}, "
            "class error (projected) : {}"
        ).format(
            i, len(dataset) / args.batch_size,
            total_contour_error['projected'] / error_count,
            total_class_error['projected'] / error_class_count
        )
    )

    print(
        (
            "[{}/{}] Binary dice (reconstructed) : {}, "
            "class dice (reconstructed) : {}"
        ).format(
            i, len(dataset) / args.batch_size,
            total_binary_dice['reconstructed'] / count,
            total_class_dice['reconstructed'] / class_count
        )
    )

    print(
        (
            "[{}/{}] Total error (reconstructed) : {}, "
            "class error (reconstructed) : {}"
        ).format(
            i, len(dataset) / args.batch_size,
            total_contour_error['reconstructed'] / error_count,
            total_class_error['reconstructed'] / error_class_count
        )
    )

    print(
        (
            "[{}/{}] Binary dice (predicted) : {}, "
            "class dice (predicted) : {}"
        ).format(
            i, len(dataset) / args.batch_size,
            total_binary_dice['predicted'] / count,
            total_class_dice['predicted'] / class_count
        )
    )

    print(
        (
            "[{}/{}] Total error (predicted) : {}, "
            "class error (predicted) : {}"
        ).format(
            i, len(dataset) / args.batch_size,
            total_contour_error['predicted'] / error_count,
            total_class_error['predicted'] / error_class_count
        )
    )

final_binary_dice = {
    k: (s / count).tolist()
    for k, s in total_binary_dice.iteritems()
}
final_class_dice = {
    k: (s / class_count).tolist()
    for k, s in total_class_dice.iteritems()
}

final_contour_error = {
    k: (s / error_count).tolist()
    for k, s in total_contour_error.iteritems()
}
final_class_error = {
    k: (s / error_class_count).tolist()
    for k, s in total_class_error.iteritems()
}

dice_scores = [
    final_binary_dice, final_class_dice
]
dice_score_file = path.join(exp_dir, "dice_scores.json")

with open(dice_score_file, "w") as f:
    json.dump(dice_scores, f)

contour_scores = [
    final_contour_error, final_class_error
]
contour_score_file = path.join(exp_dir, "contour_scores.json")
with open(contour_score_file, "w") as f:
    json.dump(contour_scores, f)