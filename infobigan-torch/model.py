# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Information maximising adversarially learned inference (InfoBiGAN)
"""
import torch
from torch import nn, optim
from pytorchure.train.trainer import Trainer
from pytorchure.utils import thumb_grid, animate_gif


def listify(item, length):
    if not (isinstance(item, tuple) or isinstance(item, list)):
        return [item] * length
    else:
        return item


class DCDiscriminator(nn.Module):
    """
    Deep convolutional discriminator network.

    Attributes
    ----------
    n_conv: int
        Number of hidden convolutional layers.
    n_fc: int
        Number of hidden fully connected layers.
    final_conv_dim
        Total number of units in the final convolutional layer.
    """
    def __init__(self,
                 channels=(1, 128, 256, 512, 1024),
                 fc=(),
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 bias=False,
                 leak=0.2):
        """Initialise a deep convolutional discriminator.

        Parameters
        ----------
        channels: tuple
            Tuple denoting number of channels in each convolutional layer.
        fc: tuple
            Tuple denoting number of hidden units in each fully connected
            hidden layer. If fully connected hidden layers are used, then the
            number of units in the first one should be set to the product of
            the number of channels in the last convolutional layer and the
            number of units in the last convolutional kernel.
        kernel_size: int or tuple
            Side length of convolutional kernel.
        stride: int or tuple
            Convolutional stride.
        padding: int or tuple
            Padding to be applied to the image during convolution.
        bias: bool or tuple
            Indicates whether each convolutional filter includes bias terms
            for each unit.
        leak: float
            Slope of the negative part of the hidden layers' leaky ReLU
            activation function.

        If any of `kernel_size`, `stride`, `padding`, or `bias` is a tuple,
        it should be exactly as long as `channels`; in this case, the ith item
        denotes the parameter value for the ith convolutional layer.
        """
        super(DCDiscriminator, self).__init__()
        self.n_conv = len(channels) + 1
        self.n_fc = len(fc)
        self.conv = nn.ModuleList()
        self.fc = nn.ModuleList()
        n_out=1

        kernel_size = listify(kernel_size, self.n_conv)
        padding = listify(padding, self.n_conv)
        stride = listify(stride, self.n_conv)
        bias = listify(bias, self.n_conv)

        self.conv.append(nn.Sequential(
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=kernel_size[0],
                stride=stride[0],
                padding=padding[0],
                bias=bias[0]
            ),
            nn.LeakyReLU(negative_slope=leak, inplace=True)
        ))
        for i, (r, s) in enumerate(zip(channels[2:], channels[1:-1])):
            j = i + 1
            self.conv.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=s,
                    out_channels=r,
                    kernel_size=kernel_size[j],
                    stride=stride[j],
                    padding=padding[j],
                    bias=bias[j]
                ),
                nn.BatchNorm2d(r),
                nn.LeakyReLU(negative_slope=leak, inplace=True)
            ))
        self.final_conv_dim = r * (kernel_size[j] ** 2)
        r = self.final_conv_dim
        for i, (r, s) in enumerate(zip(fc[1:], fc[:-1])):
            self.fc.append(nn.Sequential(
                nn.Linear(s, r),
                nn.LeakyReLU(negative_slope=leak),
                nn.BatchNorm1d(r)
            ))
        self.out = nn.Sequential(
            nn.Linear(r, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        for i in range(self.n_conv):
            x = self.conv[i](x)
        x = x.view(-1, self.final_conv_dim)
        for i in range(self.n_fc):
            x = self.fc[i](x)
        return self.out(x)


class DCGenerator(nn.Module):
    """
    Deep convolutional generator network.

    Attributes
    ----------
    n_conv: int
        Total number of hidden transpose-convolutional (deconvolutional)
        layers.
    initial_deconv_dim: tuple
        Dimensionality of the initial deconvolutional layer.
    """
    def __init__(self,
                 channels=(1024, 512, 256, 128, 1),
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 bias=False,
                 latent_dim=100):
        """Initialise a deep convolutional generator.

        Parameters
        ----------
        channels: tuple
            Tuple denoting number of channels in each deconvolutional layer.
        kernel_size: int or tuple
            Side length of convolutional kernel.
        stride: int or tuple
            Convolutional stride.
        padding: int or tuple
            Padding to be applied to the image during convolution.
        bias: bool or tuple
            Indicates whether each convolutional filter includes bias terms
            for each unit.
        latent_dim: int
            Number of latent features that the generator samples. (This is the
            number of input features.)

        If any of `kernel_size`, `stride`, `padding`, or `bias` is a tuple,
        it should be exactly as long as `channels`; in this case, the ith item
        denotes the parameter value for the ith convolutional layer.
        """
        super(DCGenerator, self).__init__()
        self.n_conv = len(channels) + 1

        kernel_size = listify(kernel_size, self.n_conv)
        padding = listify(padding, self.n_conv)
        stride = listify(stride, self.n_conv)
        bias = listify(bias, self.n_conv)
        self.initial_deconv_dim = (
            channels[0], kernel_size[0], kernel_size[0])

        self.fc = nn.Linear(latent_dim, channels[0] * (kernel_size[0]**2))
        self.conv = nn.ModuleList()
        for i, (r, s) in enumerate(zip(channels[1:-1], channels[:-2])):
            self.conv.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=s,
                    out_channels=r,
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    padding=padding[i],
                    bias=bias[i]
                ),
                nn.BatchNorm2d(r),
                nn.ReLU(inplace=True)
            ))
        self.conv.append(nn.ConvTranspose2d(
            in_channels=channels[-2],
            out_channels=channels[-1],
            kernel_size=kernel_size[-1],
            stride=stride[-1],
            padding=padding[-1],
            bias=bias[-1]
        ))
        self.out = nn.Tanh()

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, *self.initial_deconv_dim)
        for i in range(self.n_conv):
            z = self.conv[i](z)
        return self.out(z)
