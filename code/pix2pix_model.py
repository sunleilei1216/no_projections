import os
import util
import torch
import networks
import torch.nn as nn
import torchvision.models as models

from image_pool import ImagePool
from collections import OrderedDict
from torch.autograd import Variable


class Pix2PixModel(object):
    def __init__(
        self, name="experiment", phase="train", which_epoch="latest",
        batch_size=1, image_size=128, map_nc=1, input_nc=3, output_nc=3,
        num_downs=7, ngf=64, ndf=64, norm_layer="batch", pool_size=50,
        lr=0.0002, beta1=0.5, lambda_D=0.5, lambda_MSE=10,
        lambda_P=5.0, use_dropout=True, gpu_ids=[], n_layers=3,
        use_sigmoid=False, use_lsgan=True, upsampling="nearest",
        continue_train=False, checkpoints_dir="checkpoints/"
    ):
        # Define input data that will be consumed by networks
        self.input_A = torch.FloatTensor(
            batch_size, 3, image_size, image_size
        )
        self.input_map = torch.FloatTensor(
            batch_size, map_nc, image_size, image_size
        )
        norm_layer = nn.BatchNorm2d \
            if norm_layer == "batch" else nn.InstanceNorm2d

        # Define netD and netG
        self.netG = networks.UnetGenerator(
            input_nc=input_nc, output_nc=map_nc,
            num_downs=num_downs, ngf=ngf,
            use_dropout=use_dropout, gpu_ids=gpu_ids, norm_layer=norm_layer,
            upsampling_layer=upsampling
        )
        self.netD = networks.NLayerDiscriminator(
            input_nc=input_nc + map_nc, ndf=ndf,
            n_layers=n_layers, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids
        )

        # Transfer data to GPU
        if len(gpu_ids) > 0:
            self.input_A = self.input_A.cuda()
            self.input_map = self.input_map.cuda()
            self.netD.cuda()
            self.netG.cuda()

        # Initialize parameters of netD and netG
        self.netG.apply(networks.weights_init)
        self.netD.apply(networks.weights_init)

        # Load trained netD and netG
        if phase == "test" or continue_train:
            netG_checkpoint_file = os.path.join(
                checkpoints_dir, name, "netG_{}.pth".format(which_epoch)
            )
            self.netG.load_state_dict(
                torch.load(netG_checkpoint_file)
            )
            print("Restoring netG from {}".format(netG_checkpoint_file))

        if continue_train:
            netD_checkpoint_file = os.path.join(
                checkpoints_dir, name, "netD_{}.pth".format(which_epoch)
            )
            self.netD.load_state_dict(
                torch.load(netD_checkpoint_file)
            )
            print("Restoring netD from {}".format(netD_checkpoint_file))

        self.name = name
        self.gpu_ids = gpu_ids
        self.checkpoints_dir = checkpoints_dir

        # Criterions
        if phase == "train":
            self.count = 0
            self.lr = lr
            self.lambda_D = lambda_D
            self.lambda_MSE = lambda_MSE

            self.image_pool = ImagePool(pool_size)
            self.criterionGAN = networks.GANLoss(use_lsgan=use_lsgan)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()  # Landmark loss

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=self.lr, betas=(beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=self.lr, betas=(beta1, 0.999)
            )

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input_A, input_map, input_name):
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_map.resize_(input_map.size()).copy_(input_map)
        self.input_name = input_name

    def get_image_paths(self):
        return self.input_name[0]

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_map = self.netG.forward(self.real_A)
        self.real_map = Variable(self.input_map)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_map = self.netG.forward(self.real_A)
        self.real_map = Variable(self.input_map, volatile=True)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_Amap = self.image_pool.query(
            torch.cat((self.real_A, self.fake_map), 1)
        )
        self.pred_fake = self.netD.forward(fake_Amap.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_Amap = torch.cat((self.real_A, self.real_map), 1)
        self.pred_real = self.netD.forward(real_Amap)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * self.lambda_D

        self.loss_D.backward()

    def backward_G(self):
        # Third, G(A)_map = map
        self.loss_G_MSE = self.criterionMSE(
            self.fake_map, self.real_map
        ) * self.lambda_MSE

        fake_Amap = torch.cat(
            (self.real_A, self.fake_map), 1
        )
        pred_fake = self.netD.forward(fake_Amap)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G = self.loss_G_GAN + self.loss_G_MSE
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict(
            [
                ('G_GAN', self.loss_G_GAN.data[0]),
                ('G_MSE', self.loss_G_MSE.data[0]),
                ('D_real', self.loss_D_real.data[0]),
                ('D_fake', self.loss_D_fake.data[0])
            ]
        )

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_map = util.tensor2im(self.fake_map.data)
        real_map = util.tensor2im(self.real_map.data)
        return OrderedDict(
            [
                ('real_A', real_A),
                ('fake_map', fake_map),
                ('real_map', real_map)
            ]
        )

    def save(self, which_epoch):
        netD_path = os.path.join(
            self.checkpoints_dir, self.name, "netD_{}.pth".format(which_epoch)
        )
        netG_path = os.path.join(
            self.checkpoints_dir, self.name, "netG_{}.pth".format(which_epoch)
        )
        torch.save(self.netD.cpu().state_dict(), netD_path)
        torch.save(self.netG.cpu().state_dict(), netG_path)

        if len(self.gpu_ids) > 0:
            self.netG.cuda()
            self.netD.cuda()

    def update_learning_rate(self, decay):
        old_lr = self.lr
        self.lr = self.lr * decay
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = self.lr
        print('update learning rate: %f -> %f' % (old_lr, self.lr))
