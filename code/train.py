import os
import os.path as path
import re
import time
import argparse
import torch.utils as utils
from visualizer import Visualizer
from ice_dataset import ICEDataset
from pix2pix_model import Pix2PixModel

parser = argparse.ArgumentParser(
    description="Train a ice segmentation model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--name", default="2d_no_projections",
    help="the name of the experiment"
)
parser.add_argument(
    "--checkpoints_dir", default="checkpoints",
    help="a folder that contains all the trained model checkpoints"
)
parser.add_argument(
    "--data_dir", default="data/",
    help="a folder that contains all ice data"
)
parser.add_argument(
    "--batch_size", type=int, default=8,
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
    "--ngf", type=int, default=64,
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
    "--lambda_D", type=float, default=0.5,
    help="the relative importance of discrimination loss"
)
parser.add_argument(
    "--lambda_L1", type=float, default=100.0,
    help="the relative importance of L1 loss"
)
parser.add_argument(
    "--lambda_MSE", type=float, default=1000.0,
    help="the relative importance of MSE loss"
)
parser.add_argument(
    "--upsampling", default="nearest",
    help="the type of upsampling method to be used"
)
parser.add_argument(
    "--num_epochs", type=int, default=25,
    help="number of epochs to train"
)
parser.add_argument(
    "--start_epoch", type=int, default=1,
    help="the starting eporch of the training"
)
parser.add_argument(
    "--print_freq", type=int, default=160,
    help="number of batches to print current training status"
)
parser.add_argument(
    "--display_freq", type=int, default=400,
    help="number of steps to save current visualization results"
)
parser.add_argument(
    "--save_freq", type=int, default=8000000,
    help="number of steps to save current model"
)
parser.add_argument(
    "--lr", type=float, default=0.0005,
    help="learning rate"
)
parser.add_argument(
    "--lr_decay", type=float, default=0.5,
    help="learning rate decay"
)
parser.add_argument(
    "--lr_decay_freq", type=int, default=5,
    help="number of epochs to decay the learning rate"
)
parser.add_argument(
    "--beta1", type=float, default=0.5,
    help="the Adam beta1 parameter"
)
parser.add_argument(
    "--phase", default="train",
    help="the phase of the experiment"
)
parser.add_argument(
    "--continue_train", action="store_true", default=False,
    help="if specified, loading perviously saved model to resume the training"
)
parser.add_argument(
    "--which_epoch", default="latest",
    help="the epoch number used to resume the training"
)
parser.add_argument(
    "--gpu_ids", type=int, nargs="*", default=[0],
    help="the gpu devices used for training/testing"
)
args = parser.parse_args()

checkpoints_dir = os.path.join(args.checkpoints_dir, args.name)
if not os.path.isdir(checkpoints_dir):
    os.makedirs(checkpoints_dir)

pattern = re.compile("([\w_]+=[\w\.\d\'\/\[\]]+)")
options_str = str(args)
options_str = "\n".join(pattern.findall(options_str))
with open(os.path.join(checkpoints_dir, "options.txt"), "w") as f:
    f.write(options_str)
print("=============== Options ===============")
print(options_str)
print("=======================================")

dataset = ICEDataset(path.join(args.data_dir, args.phase))
data_loader = utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True
)
dataset_size = len(dataset)

model = Pix2PixModel(
    name=args.name, phase=args.phase, which_epoch=args.which_epoch,
    batch_size=args.batch_size, image_size=args.image_size,
    input_nc=3, map_nc=args.map_nc, num_downs=args.unet_size,
    ngf=args.ngf, ndf=args.ndf, norm_layer=args.norm_layer, lr=args.lr,
    beta1=args.beta1, lambda_D=args.lambda_D, lambda_MSE=args.lambda_MSE,
    gpu_ids=args.gpu_ids, upsampling=args.upsampling,
    continue_train=args.continue_train, checkpoints_dir=args.checkpoints_dir
)
visualizer = Visualizer(args.name, checkpoints_dir)

total_steps = dataset_size * (args.start_epoch - 1)
for epoch in range(args.start_epoch, args.num_epochs):
    epoch_start_time = time.time()
    epoch_iter = 0
    for data in data_loader:
        iter_start_time = time.time()
        epoch_iter += args.batch_size

        model.set_input(
            data['image'], data['segmentation_outlined'], data['image_name']
        )
        model.optimize_parameters()

        if epoch_iter % args.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / args.batch_size
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        if epoch_iter % args.display_freq == 0:
            visuals = model.get_current_visuals()
            visualizer.display_current_results(visuals, epoch)

        if epoch_iter % args.save_freq == 0:
            model.save("latest")
            print(
                "Saving current model to {}".format(checkpoints_dir)
            )

    if epoch - 1 != 0 and epoch % args.lr_decay_freq == 0:
        model.update_learning_rate(args.lr_decay)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, args.num_epochs, time.time() - epoch_start_time))
    model.save(epoch)
    print(
        "Saving current model to {}".format(checkpoints_dir)
    )
