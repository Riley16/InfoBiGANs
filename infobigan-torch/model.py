# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
InfoBiGAN
~~~~~~~~~
Information maximising adversarially learned inference network
"""
import torch
from torch import nn #, optim
#from pytorchure.train.trainer import Trainer
#from pytorchure.utils import thumb_grid, animate_gif


eps = 1e-15


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
        size = (size - k + 2 * p) // s + 1
    return int(size)


class InfoBiGAN(object):
    """An information maximising adversarially learned inference network
    (InfoBiGAN).

    Attributes
    ----------
    discriminator: DualDiscriminator
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
    latent_dim: int
        Dimensionality of the latent space. Currently, this is basically a
        vanilla BiGAN, so this only includes noise.
    """
    def __init__(self,
                 channels=(1, 128, 256, 512, 1024),
                 kernel_size=4,
                 stride=2,
                 padding=(3, 1, 1, 1),
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

        self.discriminator = DualDiscriminator(
            channels=channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias, manifest_dim=manifest_dim,
            latent_dim=latent_dim)
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

    def zero_grad(self):
        self.encoder.zero_grad()
        self.generator.zero_grad()
        self.discriminator.zero_grad()

    def load_state_dict(self, params_g, params_e,
                        params_d_z, params_d_x, params_d_xz):
        self.encoder.load_state_dict(params_e)
        self.generator.load_state_dict(params_g)
        self.discriminator.load_state_dict(params_z=params_d_z,
                                           params_x=params_d_x,
                                           params_xz=params_d_xz)


class DualDiscriminator(nn.Module):
    """A discriminator network that learns to identify whether a (latent,
    manifest) pair is drawn from the encoder or from the decoder.

    Attributes
    ----------
    x_discriminator: DCNetwork
        Representational network for manifest-space data.
    z_discriminator: DCNetwork
        Representational network for latent-space data.
    zx_discriminator: DCNetwork
        Discriminator that splices together representations of latent- and
        manifest-space data and yields a decision regarding the provenance
        of the data pair.
    regulariser: QStack
        . . . Not yet implemented . . .
    """
    def __init__(self,
                 manifest_dim=28,
                 latent_dim=100,
                 channels=(1, 128, 256, 512, 1024),
                 kernel_size=4,
                 stride=2,
                 padding=(3, 1, 1, 1),
                 bias=False):
        """Initialise a dual discriminator.

        Parameters
        ----------
        manifest_dim: int
            Side length of the input image.
        latent_dim: int
            Dimensionality of the latent space.
        channels: tuple
            Tuple denoting number of channels in each convolutional layer of
            the manifest-space representational network.
        kernel_size: int or tuple
            Side length of convolutional kernel in the manifest-space
            representational network.
        stride: int or tuple
            Convolutional stride for the manifest-space representational
            network.
        padding: int or tuple
            Padding to be applied to the manifest-space image data during
            convolution.
        bias: bool or tuple
            Indicates whether each convolutional filter in the image
            (manifest) representational network includes bias terms for each
            unit.
        """
        super(DualDiscriminator, self).__init__()
        self.x_discriminator = DCNetwork(
            channels=channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias, in_dim=manifest_dim,
            out_dim=latent_dim*2, final_act='leaky', embedded=(False, True))
        self.z_discriminator = DCNetwork(
            channels=(latent_dim), fc=(latent_dim*2, latent_dim*2),
            kernel_size=1, stride=1, padding=0, bias=True, in_dim=1,
            out_dim=latent_dim*2, final_act='leaky', embedded=(False, True))
        self.zx_discriminator = DCNetwork(
            channels=(latent_dim*4), fc=(latent_dim*4, latent_dim*4),
            kernel_size=1, stride=1, padding=0, bias=True, in_dim=1,
            out_dim=1, embedded=(True, False))

    def train(self):
        self.z_discriminator.train()
        self.x_discriminator.train()
        self.zx_discriminator.train()

    def eval(self):
        self.z_discriminator.eval()
        self.x_discriminator.eval()
        self.zx_discriminator.eval()

    def load_state_dict(self, params_z, params_x, params_zx):
        self.z_discriminator.load_state_dict(params_z)
        self.x_discriminator.load_state_dict(params_x)
        self.zx_discriminator.load_state_dict(params_zx)

    def forward(self, z, x):
        z = self.z_discriminator(z)
        x = self.x_discriminator(x)
        zx = torch.cat([z, x], 1) + eps
        zx = self.zx_discriminator(zx)
        return zx



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
                 padding=(3, 1, 1, 1, 0),
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
        self.conv = nn.ModuleList()
        try:
            channels = list(channels)
            self.n_conv = len(channels) - 1
        except TypeError:
            channels = [channels]
            self.n_conv = len(channels) - 1

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
                 padding=(3, 1, 1, 1),
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
        try:
            channels = list(channels)
            n_conv = len(channels) - 1
        except TypeError:
            channels = [channels]
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
                 padding=(1, 1, 1, 3),
                 bias=False,
                 final_act='tanh',
                 embedded=(False, False),
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
        try:
            channels = list(channels)
            n_conv = len(channels) - 1
        except TypeError:
            channels = [channels]
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


class QStack(nn.Module):
    """Q distribution neural network for informational regularisation.

    Attributes
    ----------
    q_regularised: ModuleDict
        Dictionary of modules that yield distributional parameters for the
        compressible input c.
    latent_categorical: int
        Number of regularised categorical variables in the latent space.
    latent_gaussian: int
        Number of regularised Gaussian variables in the latent space.
    """
    def __init__(self,
                 latent_categorical,
                 latent_gaussian,
                 hidden_dim=100):
        """Initialise a Q stack.

        Parameters
        ----------
        latent_categorical: tuple
            List of level counts for uniformly categorically distributed
            variables in c. For instance, (10, 8) denotes one variable with 10
            levels and another with 8 levels.
        latent_gaussian: int
            Number of normally distributed variables in c.
        hidden_dim: int
            Dimensionality of the hidden layer.
        """
        super(QStack, self).__init__()
        self.q_input = DCNetwork(
            channels=(hidden_dim),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            leak=0.2,
            final_act='leaky',
            in_dim=1,
            out_dim=hidden_dim
        )
        self.q_regularised = nn.ModuleDict()
        self.latent_gaussian = latent_gaussian
        self.latent_categorical = len(latent_categorical)
        for i, levels in enumerate(latent_categorical):
            self.q_regularised['cat{}'.format(i)] = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=levels, kernel_size=1,
                    stride=1, padding=0, bias=True),
                nn.Softmax()
            )
        if latent_gaussian > 0:
            self.q_regularised['gaussian'] = nn.ModuleDict({
                'mean': nn.Conv2d(
                    in_channels=hidden_dim, out_channels=latent_gaussian,
                    kernel_size=1, stride=1, padding=0, bias=True),
                'logstd': nn.Conv2d(
                    in_channels=hidden_dim, out_channels=latent_gaussian,
                    kernel_size=1, stride=1, padding=0, bias=True)
            })

    def forward(self, x):
        c = {}
        x = self.q_input(x)
        for i in range(self.latent_categorical):
            c['cat{}'.format(i)] = self.q_regularised['cat{}'.format(i)](x)
        if self.latent_gaussian > 0:
            c['gaussian'] = {}
            c['gaussian']['mean'] = self.q_regularised['gaussian']['mean'](x)
            c['gaussian']['logstd'] = (
                self.q_regularised['gaussian']['logstd'](x))
        return c
