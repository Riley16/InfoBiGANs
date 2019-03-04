# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Convolutional neural networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Interfaces for building simple convolutional neural networks.
"""
import torch
from torch import nn
from .utils import _listify


def _conv_out_size(input_size, kernels, strides, paddings):
    """Get the output size of a convolutional stack with the specified
    parameters.
    """
    size = input_size
    for k, s, p in zip(kernels, strides, paddings):
        size = (size - k + 2 * p) // s + 1
    return int(size)


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
                 padding=(2, 2, 1, 1, 0),
                 bias=(False, False, False, False, True),
                 leak=0.2):
        """Initialise a deep convolutional network.

        Parameters
        ----------
        channels: tuple
            Tuple denoting number of channels in each convolutional layer. Set
            the final element to 1 for discriminator behaviour. Set to any
            other value for general ConvNet behaviour.
        hidden: str or tuple
            Specifies the hidden layer type.
            * `conv` denotes a standard convolutional layer.
            * `transpose` denotes a transpose convolutional layer.
            * `maxpool` denotes a max pooling layer. The input `channels` must
              equal the output `channels.`
        nonlinearity: 'leaky' or 'relu' or 'sigmoid' or 'tanh' or tuple
            Nonlinearity to use in each convolutional layer.
        kernel_size: int or tuple
            Side length of convolutional kernel. A convolutional layer whose
            kernel size is equivalent to the entire field of the previous
            convolutional layer is effectively equivalent to a fully connected
            layer if there is no padding.
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
        channels = _listify(channels)
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
    """
    def __init__(self,
                 channels=(1, 128, 256, 512, 1024),
                 fc=(),
                 kernel_size=4,
                 stride=2,
                 padding=(2, 2, 1, 1),
                 bias=False,
                 leak=0.2,
                 final_act='sigmoid',
                 embedded=(False, False),
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
        final_act: `sigmoid` or `tanh` or `relu` or `leaky` or None
            Nonlinear activation function for the final layer of the network.
        embedded: (bool, bool)
            Indicates whether the network is embedded in a larger network.
            The first indicator should be True if the network is directly
            receiving input from another, while the second indicator should be
            True if the network is directly passing output to another.
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
        channels = _listify(channels)
        n_conv = len(channels) - 1

        fc = list(fc)
        n_fc = len(fc) + 1
        conv_out_size = _conv_out_size(in_dim,
                                       _listify(kernel_size, n_conv),
                                       _listify(stride, n_conv),
                                       _listify(padding, n_conv))

        kernel_size = (_listify(kernel_size, n_conv) + [conv_out_size]
                       + [1] * (n_fc - 1))
        channels = channels + fc + [out_dim]
        padding = _listify(padding, n_conv) + [0] * n_fc
        stride = _listify(stride, n_conv) + [1] * n_fc
        bias = _listify(bias, n_conv) + [True] * n_fc
        nonlinearity = ['leaky'] * (n_conv + n_fc - 1) + [final_act]
        batch_norm = ([embedded[0]] + [True] * (n_conv + n_fc - 2)
                      + [embedded[1]])

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
                 padding=(1, 1, 2, 2),
                 bias=False,
                 final_act='tanh',
                 embedded=(False, False),
                 latent_dim=100,
                 target_dim=28,
                 reg_categorical=(10,),
                 reg_gaussian=2):
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
        final_act: `tanh` or `sigmoid` or `relu` or `leaky` or None
            Nonlinear activation function for the final layer of the network.
        embedded: (bool, bool)
            Indicates whether the network is embedded in a larger network.
            The first indicator should be True if the network is directly
            receiving input from another, while the second indicator should be
            True if the network is directly passing output to another.
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
        channels = _listify(channels)
        n_conv = len(channels) - 1
        fc = list(fc)
        n_fc = len(fc) + 1

        conv_in_size = _conv_out_size(target_dim,
                                      _listify(kernel_size, n_conv)[::-1],
                                      _listify(stride, n_conv)[::-1],
                                      _listify(padding, n_conv)[::-1])

        channels = [latent_dim] + fc + channels
        kernel_size = ([1] * (n_fc - 1) + [conv_in_size]
                       + _listify(kernel_size, n_conv))
        padding = [conv_in_size - 1] * n_fc + _listify(padding, n_conv)
        stride = [1] * n_fc + _listify(stride, n_conv)
        nonlinearity = [None] + ['relu'] * (n_conv + n_fc - 2) + [final_act]
        hidden = ['conv'] * n_fc + ['transpose'] * n_conv
        bias = [True] * n_fc + _listify(bias, n_conv) + [False]
        batch_norm = ([embedded[0]] + [True] * (n_conv + n_fc - 2)
                      + [embedded[1]])

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
