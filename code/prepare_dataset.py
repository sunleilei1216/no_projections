import os
import os.path as path
import json
import random
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(
    description="Prepare 2D ICE dataset for training",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--image_dir", default="data/usd_images",
    help="a folder that contains all the 2D ICE images"
)
parser.add_argument(
    "--contour_dir", default="data/ann_contours",
    help="a folder that contains all the annotated ICE contours"
)
parser.add_argument(
    "--output_dir", default="data/",
    help="a folder that stores all the prepared ICE data"
)
parser.add_argument(
    "--num_organs", type=int, default=6,
    help="the maximum possible number of organs in an ICE image"
)
parser.add_argument(
    "--image_size", type=int, default=256,
    help="the image size of the prepared ICE images"
)
parser.add_argument(
    "--train_proportion", type=float, default=0.9,
    help="the proportion of the training images"
)
args = parser.parse_args()


def resize_image(image, size):
    """The size of the original image may not be the same as the required
    size of the network. We need to resize it so that it can fit the network.
    """
    image_shape = image.shape[:2]
    min_dim = np.argmin(image_shape)
    if min_dim == 0:
        start = (image_shape[1] - image_shape[0]) / 2
        image_cropped = image[:, start:start+image_shape[0]]
    else:
        start = (image_shape[0] - image_shape[1]) / 2
        image_cropped = image[start:start+image_shape[1], :]

    image_obj = Image.fromarray(image_cropped)
    image_obj.thumbnail((size, size), Image.ANTIALIAS)
    return np.array(image_obj)

contour_files = [
    path.join(args.contour_dir, f)
    for f in os.listdir(args.contour_dir)
    if f.endswith(".npy")
]

# associate image names with image file and contour files
image_name2files = {}
subject_name2image_names = {}
for contour_file in contour_files:
    contour_name = path.split(contour_file)[-1][:-4]
    split_items = contour_name.split("_")
    subject_name = "_".join(split_items[:2])
    frame_name = "_".join(split_items[2:-2])
    image_name = subject_name + "_" + frame_name
    organ_id = int(split_items[-2])

    if subject_name not in subject_name2image_names:
        subject_name2image_names[subject_name] = set()
    subject_name2image_names[subject_name].add(image_name)

    if image_name not in image_name2files:
        image_file = path.join(
            args.image_dir, subject_name, frame_name + ".bmp"
        )
        segmentation_file = path.join(
            args.image_dir, subject_name, frame_name + "_segmentation.npy"
        )
        reconstructed_file = path.join(
            args.image_dir, subject_name, frame_name + "_reconstructed.npy"
        )
        mask_file = path.join(
            args.image_dir, subject_name, frame_name + "_mask.npy"
        )
        image_name2files[image_name] = {
            "image_file": image_file,
            "segmentation_file": segmentation_file,
            "reconstructed_file": reconstructed_file,
            "mask_file": mask_file,
            "contour_files": [[]] * 6
        }
    this_contour_files = image_name2files[image_name]["contour_files"]
    this_contour_files[organ_id] = contour_file

subject_names = subject_name2image_names.keys()
train_subject_list_file = path.join(args.output_dir, "train.json")
test_subject_list_file = path.join(args.output_dir, "test.json")
train_subject_2d_list_file = path.join(args.output_dir, "train_2d.json")
test_subject_2d_list_file = path.join(args.output_dir, "test_2d.json")
with open(train_subject_list_file) as f:
    train_subject_names = list(set(json.load(f)) & set(subject_names))
with open(test_subject_list_file) as f:
    test_subject_names = list(set(json.load(f)) & set(subject_names))
with open(train_subject_2d_list_file, "w") as f:
    json.dump(train_subject_names, f)
with open(test_subject_2d_list_file, "w") as f:
    json.dump(test_subject_names, f)

train_dir = path.join(args.output_dir, "train")
test_dir = path.join(args.output_dir, "test")
if not path.isdir(train_dir):
    os.makedirs(train_dir)
if not path.isdir(test_dir):
    os.makedirs(test_dir)

image_index = 0
for image_name, data_files in image_name2files.iteritems():
    print("[{}/{}] Preparing data for {}".format(
        image_index, len(image_name2files), image_name
    ))
    image_index += 1

    subject_name = "_".join(image_name.split("_")[:2])
    if subject_name in train_subject_names:
        phase_dir = train_dir
    elif subject_name in test_subject_names:
        phase_dir = test_dir
    else:
        continue

    phase_dir = (
        train_dir if subject_name in train_subject_names else test_dir
    )

    image = np.array(Image.open(data_files["image_file"]))
    mask = np.load(data_files["mask_file"])
    hardness = np.zeros(args.num_organs).astype(int)

    segmentation_projected = (np.load(
        data_files["segmentation_file"]
    ) > 0.5).astype(np.uint8)
    segmentation_reconstructed = (np.load(
        data_files["reconstructed_file"]
    ) > 0.5).astype(np.uint8)
    segmentation_outlined = np.zeros(
        (args.num_organs,) + image.shape[:2]
    ).astype(np.uint8)
    for contour_index, contour_file in enumerate(data_files["contour_files"]):
        if contour_file:
            contour_dict_file = contour_file[:-4] + ".json"
            with open(contour_dict_file) as f:
                contour_dict = json.load(f)
            if "complexity" in contour_dict:
                complexity = contour_dict["complexity"]
                contour = np.load(contour_file)
                if complexity == "Easy":
                    segmentation_outlined[
                        contour_index
                    ] = contour.astype(np.float32)
                elif complexity == "Hard":
                    hardness[contour_index] = 1

                    max_cp_with_p = 0
                    max_cp_index_with_p = -1
                    for spline_index, spline in enumerate(
                        contour_dict["splines"]
                    ):
                        if "p" in spline and len(spline['p']) > 0:
                            if len(spline['cp']) > max_cp_with_p:
                                max_cp_with_p = len(spline['cp'])
                                max_cp_index_with_p = spline_index

                    if max_cp_index_with_p < 0:
                        segmentation_outlined[
                            contour_index
                        ] = segmentation_projected[contour_index]
                    else:
                        segmentation_outlined[
                            contour_index
                        ] = contour.astype(np.float32)

    image = (image.transpose(2, 0, 1) * mask).transpose(1, 2, 0)
    segmentation_projected = segmentation_projected * mask
    segmentation_reconstructed = segmentation_reconstructed * mask
    segmentation_outlined = segmentation_outlined * mask
    organs = (segmentation_outlined.sum((1, 2)) > 0).astype(int)

    image = resize_image(image, args.image_size)
    mask = resize_image(mask, args.image_size)
    segmentation_projected = np.array([
        resize_image(s, args.image_size)
        for s in segmentation_projected
    ])
    segmentation_reconstructed = np.array([
        resize_image(s, args.image_size)
        for s in segmentation_reconstructed
    ])
    segmentation_outlined = np.array([
        resize_image(s, args.image_size)
        for s in segmentation_outlined
    ])
    label = {"hardness": hardness.tolist(), "organs": organs.tolist()}

    out_image_file = path.join(phase_dir, image_name + ".jpg")
    out_mask_file = path.join(phase_dir, image_name + "_mask.npy")
    out_segmentation_projected_file = path.join(
        phase_dir, image_name + "_segmentation_projected.npy"
    )
    out_segmentation_reconstructed_file = path.join(
        phase_dir, image_name + "_segmentation_reconstructed.npy"
    )
    out_segmentation_outlined_file = path.join(
        phase_dir, image_name + "_segmentation_outlined.npy"
    )
    out_label_file = path.join(phase_dir, image_name + "_label.json")

    Image.fromarray(image).save(out_image_file)
    np.save(out_mask_file, mask.astype(bool))
    np.save(
        out_segmentation_projected_file, segmentation_projected.astype(bool)
    )
    np.save(
        out_segmentation_reconstructed_file,
        segmentation_reconstructed
    )
    np.save(
        out_segmentation_outlined_file, segmentation_outlined.astype(bool)
    )
    with open(out_label_file, "w") as f:
        json.dump(label, f)

    for organ_index, organ in enumerate(organs):
        if organ != 0:
            output_segmentation_projected_organ_file = path.join(
                phase_dir,
                "{}_segmentation_projected_{}.jpg".format(
                    image_name, organ_index
                )
            )
            output_segmentation_reconstructed_organ_file = path.join(
                phase_dir,
                "{}_segmentation_reconstructed_{}.jpg".format(
                    image_name, organ_index
                )
            )
            output_segmentation_outlined_organ_file = path.join(
                phase_dir,
                "{}_segmentation_outlined_{}.jpg".format(
                    image_name, organ_index
                )
            )
            Image.fromarray(
                segmentation_projected[organ_index] * 255
            ).convert("RGB").save(output_segmentation_projected_organ_file)
            Image.fromarray(
                segmentation_reconstructed[organ_index] * 255
            ).convert("RGB").save(output_segmentation_reconstructed_organ_file)
            Image.fromarray(
                segmentation_outlined[organ_index] * 255
            ).convert("RGB").save(output_segmentation_outlined_organ_file)
