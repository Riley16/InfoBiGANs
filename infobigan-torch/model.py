# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Information maximising adversarially learned inference (InfoBiGAN)
"""
import torch
from torch import nn #, optim
import numpy as np
#from pytorchure.train.trainer import Trainer
#from pytorchure.utils import thumb_grid, animate_gif


def _listify(item, length):
    if not (isinstance(item, tuple) or isinstance(item, list)):
        return [item] * length
    else:
        return list(item)


def _conv_out_size(input_size, kernels, strides, paddings):
    """Get the output size of a convolutional stack with the specified
    parameters.
    """
    size = input_size
    for k, s, p in zip(kernels, strides, paddings):
        size = np.floor((size - k + 2 * p) / s + 1)
    return int(size)


class InfoBiGAN(object):
    """An information maximising adversarially learned inference network
    (InfoBiGAN).

    Attributes
    ----------
    discriminator: DCNetwork
        InfoBiGAN's discriminator network, which is presented a set of
        latent space-manifest space pairs and determines whether each
        pair was produced by the encoder or the generator.
    generator: DCTranspose
        InfoBiGAN's generator network, which learns the underlying
        distribution of a dataset through a minimax game played against the
        discriminator.
    encoder: DCNetwork
        InfoBiGAN's inferential network, which learns the latent space
        encodings of a dataset through a minimax game played against the
        discriminator.
    regulariser: QLayer
        . . . Not yet implemented . . .
    latent_dim: int
        Dimensionality of the latent space. Currently, this is basically a
        vanilla BiGAN, so this only includes noise.
    """
    def __init__(self,
                 channels=(1, 128, 256, 512, 1024),
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 bias=False,
                 manifest_dim=28,
                 latent_dim=100):
        """Initialise an information maximising adversarially learned
        inference network (InfoBiGAN).

        Parameters are ordered according to the convolutional networks
        (discriminator and encoder). For instance, the second channel
        parameter denotes the number of channels in the second convolutional
        layer. The transpose-convolutional network (generator) currently
        uses an inverse architecture, so that the same parameter denotes the
        number of channels in its second-to-last deconvolutional layer.

        Currently, the encoder and discriminator are initialised with the
        same base architecture; however, the discriminator has a single output
        unit while the encoder has a number of output units equal to the
        dimensionality of the latent space.

        Parameters
        ----------
        channels: tuple
            Tuple denoting number of channels in each convolutional layer.
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
            Number of latent features that the generator network samples.
        manifest_dim: int
            Side length of the input image.

        If any of `kernel_size`, `stride`, `padding`, or `bias` is a tuple,
        it should be exactly as long as `channels`; in this case, the ith item
        denotes the parameter value for the ith convolutional layer.
        """
        n_conv = len(channels) + 1
        kernel_size = _listify(kernel_size, n_conv)
        padding = _listify(padding, n_conv)
        stride = _listify(stride, n_conv)
        bias = _listify(bias, n_conv)

        self.discriminator = DCNetwork(
            channels=channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias, in_dim=manifest_dim, out_dim=1)
        self.encoder = DCNetwork(
            channels=channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias, in_dim=manifest_dim,
            out_dim=latent_dim)
        self.generator = DCTranspose(
            channels=channels[::-1], kernel_size=kernel_size[::-1],
            stride=stride[::-1], padding=padding[::-1], bias=bias[::-1],
            latent_dim=latent_dim, target_dim=manifest_dim)
        self.latent_dim = latent_dim
        self.manifest_dim = manifest_dim

    def train(self):
        self.discriminator.train()
        self.generator.train()
        self.encoder.train()

    def eval(self):
        self.discriminator.eval()
        self.generator.eval()
        self.encoder.eval()

    def load_state_dict(self, params_d, params_g, params_e):
        self.discriminator.load_state_dict(params_d)
        self.generator.load_state_dict(params_g)
        self.encoder.load_state_dict(params_e)


class DCArchitecture(nn.Module):
    """
    Generalised deep convolutional network architecture.

    Attributes
    ----------
    n_conv: int
        Number of hidden convolutional layers.
    """
    def __init__(self,
                 channels=(1, 128, 256, 512, 1024, 1),
                 hidden='conv',
                 nonlinearity=('leaky', 'leaky', 'leaky', 'leaky', 'sigmoid'),
                 kernel_size=(4, 4, 4, 4, 1),
                 batch_norm=(False, True, True, True, False),
                 stride=(2, 2, 2, 2, 1),
                 padding=(1, 1, 1, 1, 0),
                 bias=(False, False, False, False, True),
                 leak=0.2):
        """Initialise a deep convolutional network.

        Parameters
        ----------
        channels: tuple
            Tuple denoting number of channels in each convolutional layer. Set
            the final element to 1 for discriminator behaviour. Set to any
            other value for general ConvNet behaviour.
        hidden: bool or tuple
            Specifies the hidden layer type.
            * `conv` denotes a standard convolutional layer.
            * `transpose` denotes a transpose convolutional layer.
            * `maxpool` denotes a max pooling layer. The input `channels` must
              equal the output `channels.`
        nonlinearity: 'leaky' or 'relu' or 'sigmoid' or 'tanh' or tuple
            Nonlinearity to use in each convolutional layer.
        kernel_size: int or tuple
            Side length of convolutional kernel. A side length of 1 yields a
            convolutional layer that is effectively equivalent to a fully
            connected layer.
        batch_norm: bool or tuple
            Indicates whether batch normalisation should be applied to each
            layer.
        stride: int or tuple
            Convolutional stride.
        padding: int or tuple
            Padding to be applied to the image during convolution.
        bias: bool or tuple
            Indicates whether each convolutional filter includes bias terms
            for each unit.
        leak: float
            Slope of the negative part of the hidden layers' leaky ReLU
            activation function. Used only if a leaky ReLU activation function
            is specified.

        If any of `hidden`, `nonlinearity`, `kernel_size`, `batch_norm`,
        `stride`, `padding`, or `bias` is a tuple, it should be 1 shorter than
        `channels`; in this case, the ith item denotes the parameter value for
        the ith convolutional layer (mapping the ith index of channels to the
        (i + 1)th index of channels).
        """
        super(DCArchitecture, self).__init__()
        self.n_conv = len(channels) - 1
        self.conv = nn.ModuleList()

        nonlinearity = _listify(nonlinearity, self.n_conv)
        kernel_size = _listify(kernel_size, self.n_conv)
        batch_norm = _listify(batch_norm, self.n_conv)
        padding = _listify(padding, self.n_conv)
        hidden = _listify(hidden, self.n_conv)
        stride = _listify(stride, self.n_conv)
        bias = _listify(bias, self.n_conv)

        for i, (r, s) in enumerate(zip(channels[1:], channels[:-1])):
            layer = []

            if hidden[i] == 'conv':
                layer.append(
                    nn.Conv2d(
                        in_channels=s,
                        out_channels=r,
                        kernel_size=kernel_size[i],
                        stride=stride[i],
                        padding=padding[i],
                        bias=bias[i]
                    ))
            elif hidden[i] == 'transpose':
                layer.append(
                    nn.ConvTranspose2d(
                        in_channels=s,
                        out_channels=r,
                        kernel_size=kernel_size[i],
                        stride=stride[i],
                        padding=padding[i],
                        bias=bias[i]
                    ))
            elif hidden[i] == 'maxpool':
                layer.append(
                    nn.MaxPool2d(
                        kernel_size=kernel_size[i],
                        stride=stride[i],
                        padding=padding[i],
                        bias=bias[i]
                    ))

            if batch_norm[i]:
                layer.append(nn.BatchNorm2d(r))

            if nonlinearity[i] == 'leaky':
                layer.append(nn.LeakyReLU(negative_slope=leak, inplace=True))
            elif nonlinearity[i] == 'relu':
                layer.append(nn.ReLU(inplace=True))
            elif nonlinearity[i] == 'sigmoid':
                layer.append(nn.Sigmoid())
            elif nonlinearity[i] == 'tanh':
                layer.append(nn.Tanh())

            self.conv.append(nn.Sequential(*layer))

        self._final_conv_dim = r * (kernel_size[i] ** 2)

    def forward(self, x):
        for i in range(self.n_conv):
            x = self.conv[i](x)
        return x


class DCNetwork(DCArchitecture):
    """
    Deep convolutional network.

    Attributes
    ----------
    n_conv: int
        Number of hidden convolutional layers.
    n_fc: int
        Number of hidden fully connected layers.
    """
    def __init__(self,
                 channels=(1, 128, 256, 512, 1024),
                 fc=(),
                 kernel_size=4,
                 stride=2,
                 padding=(3, 1, 1, 1),
                 bias=False,
                 leak=0.2,
                 in_dim=28,
                 out_dim=1):
        """Initialise a deep convolutional network.

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
        in_dim: int
            Input pixel dimensionality. Currently assumes a square input.
            This is the width or height of the input, not the number of
            channels or the batch cardinality.
        out_dim: int
            Number of output units. Set to 1 for discriminator behaviour.
            Set to any other value for general ConvNet behaviour.

        If any of `kernel_size`, `stride`, `padding`, or `bias` is a tuple,
        it should be exactly as long as `channels`; in this case, the ith item
        denotes the parameter value for the ith convolutional layer.
        """
        n_conv = len(channels) - 1

        fc = list(fc)
        n_fc = len(fc) + 1
        conv_out_size = _conv_out_size(in_dim,
                                       _listify(kernel_size, n_conv),
                                       _listify(stride, n_conv),
                                       _listify(padding, n_conv))

        kernel_size = (_listify(kernel_size, n_conv) + [conv_out_size]
                       + [1] * (n_fc - 1))
        channels = _listify(channels, n_conv) + fc + [out_dim]
        padding = _listify(padding, n_conv) + [0] * n_fc
        stride = _listify(stride, n_conv) + [1] * n_fc
        bias = _listify(bias, n_conv) + [True] * n_fc
        nonlinearity = ['leaky'] * (n_conv + n_fc - 1) + ['sigmoid']
        batch_norm = [False] + [True] * (n_conv + n_fc - 2) + [False]

        super(DCNetwork, self).__init__(
            channels=channels,
            kernel_size=kernel_size,
            hidden='conv',
            nonlinearity=nonlinearity,
            batch_norm=batch_norm,
            stride=stride,
            padding=padding,
            bias=bias,
            leak=leak
        )


class DCTranspose(DCArchitecture):
    """
    Deep convolutional generator network.

    Attributes
    ----------
    n_conv: int
        Total number of hidden transpose-convolutional (deconvolutional)
        layers.
    """
    def __init__(self,
                 channels=(1024, 512, 256, 128, 1),
                 fc=(),
                 kernel_size=4,
                 stride=2,
                 padding=(1, 1, 1, 3),
                 bias=False,
                 latent_dim=100,
                 target_dim=28):
        """Initialise a deep convolutional generator.

        Parameters
        ----------
        channels: tuple
            Tuple denoting number of channels in each deconvolutional layer.
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
        latent_dim: int
            Number of latent features that the generator samples. (This is the
            number of input features.)
        target_dim: int
            Desired dimensionality of the image generated by the network. This
            is the side length of the image in pixels.

        If any of `kernel_size`, `stride`, `padding`, or `bias` is a tuple,
        it should be exactly as long as `channels`; in this case, the ith item
        denotes the parameter value for the ith convolutional layer.
        """
        n_conv = len(channels) - 1

        fc = list(fc)
        n_fc = len(fc) + 1
        conv_in_size = _conv_out_size(target_dim,
                                      _listify(kernel_size, n_conv)[::-1],
                                      _listify(stride, n_conv)[::-1],
                                      _listify(padding, n_conv)[::-1])

        channels = [latent_dim] + fc + _listify(channels, n_conv)
        kernel_size = ([1] * (n_fc - 1) + [conv_in_size]
                       + _listify(kernel_size, n_conv))
        padding = [conv_in_size - 1] * n_fc + _listify(padding, n_conv)
        stride = [1] * n_fc + _listify(stride, n_conv)
        nonlinearity = [None] + ['relu'] * (n_conv + n_fc - 2) + ['tanh']
        hidden = ['conv'] * n_fc + ['transpose'] * n_conv
        bias = [True] * n_fc + _listify(bias, n_conv) + [False]
        batch_norm = [False] + [True] * (n_conv + n_fc - 2) + [False]

        super(DCTranspose, self).__init__(
            channels=channels,
            kernel_size=kernel_size,
            hidden=hidden,
            nonlinearity=nonlinearity,
            batch_norm=batch_norm,
            stride=stride,
            padding=padding,
            bias=bias
        )

