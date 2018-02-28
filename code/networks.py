import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find(
        'InstanceNorm2d'
    ) != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, extra_nc=1, num_downs=7, ngf=64,
            norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[],
            upsampling_layer="deconv"
    ):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                upsampling_layer=upsampling_layer
            )
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer,
            upsampling_layer=upsampling_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer,
            upsampling_layer=upsampling_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, unet_block, norm_layer=norm_layer,
            upsampling_layer=upsampling_layer
        )
        unet_block = UnetSkipConnectionBlock(
            input_nc, ngf, unet_block,
            outermost=True, norm_layer=norm_layer, final_nc=output_nc,
            upsampling_layer=upsampling_layer
        )

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self, outer_nc, inner_nc,
        submodule=None, outermost=False,
        innermost=False, norm_layer=nn.BatchNorm2d,
        use_dropout=False, final_nc=1, upsampling_layer="deconv"
    ):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            if upsampling_layer == "deconv":
                upconv = nn.ConvTranspose2d(
                    inner_nc * 2, final_nc,
                    kernel_size=4, stride=2, padding=1
                )
                up = [uprelu, upconv, nn.Tanh()]
            elif upsampling_layer == "nearest":
                upsample = nn.Upsample(scale_factor=2)
                upconv = nn.Conv2d(
                    inner_nc * 2, final_nc,
                    kernel_size=3, stride=1, padding=1
                )
                up = [uprelu, upsample, upconv, nn.Tanh()]
            else:
                raise LookupError("Invalid upsampling layer")

            down = [downconv]
            model = down + [submodule] + up
        elif innermost:
            if upsampling_layer == "deconv":
                upconv = nn.ConvTranspose2d(
                    inner_nc, outer_nc, kernel_size=4, stride=2, padding=1
                )
                up = [uprelu, upconv, upnorm]
            elif upsampling_layer == "nearest":
                upsample = nn.Upsample(scale_factor=2)
                upconv = nn.Conv2d(
                    inner_nc, outer_nc,
                    kernel_size=3, stride=1, padding=1
                )
                up = [uprelu, upsample, upconv, upnorm]
            else:
                raise LookupError("Invalid upsampling layer")

            down = [downrelu, downconv]
            model = down + up
        else:
            if upsampling_layer == "deconv":
                upconv = nn.ConvTranspose2d(
                    inner_nc * 2, outer_nc, kernel_size=4,
                    stride=2, padding=1
                )
                up = [uprelu, upconv, upnorm]
            elif upsampling_layer == "nearest":
                upsample = nn.Upsample(scale_factor=2)
                upconv = nn.Conv2d(
                    inner_nc * 2, outer_nc,
                    kernel_size=3, stride=1, padding=1
                )
                up = [uprelu, upsample, upconv, upnorm]
            else:
                raise LookupError("Invalid upsampling layer")

            down = [downrelu, downconv, downnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(
        self, input_nc, ndf=64, n_layers=3,
        norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]
    ):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=2, padding=padw
                ),
                # TODO: use InstanceNorm
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=1, padding=padw
            ),
            # TODO: use InstanceNorm
            norm_layer(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(
            input.data, torch.cuda.FloatTensor
        ):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(
        self, use_lsgan=True,
        target_real_label=1.0, target_fake_label=0.0
    ):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        tensor_fn = torch.cuda.FloatTensor \
            if "cuda" in input.data.type() else torch.FloatTensor
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = tensor_fn(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False
                )
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = tensor_fn(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False
                )
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
