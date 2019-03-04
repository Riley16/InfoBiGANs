# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
InfoBiGAN
~~~~~~~~~
Information maximising adversarially learned inference network
"""
import torch
from torch import nn
from .conv import DCTranspose, DCNetwork
from .utils import eps, _listify


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
    encoder: RegularisedEncoder
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
                 padding=(2, 2, 1, 1),
                 bias=False,
                 manifest_dim=28,
                 latent_dim=100,
                 reg_categorical=(10,),
                 reg_gaussian=2):
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
        reg_categorical: tuple
            List of level counts for uniformly categorically distributed
            variables in the latent space. For instance, (10, 8) denotes one
            variable with 10 levels and another with 8 levels.
        reg_gaussian: int
            Number of regularised normally distributed variables in the latent
            space.

        If any of `kernel_size`, `stride`, `padding`, or `bias` is a tuple,
        it should be exactly as long as `channels`; in this case, the ith item
        denotes the parameter value for the ith convolutional layer.
        """
        n_conv = len(channels) + 1
        kernel_size = _listify(kernel_size, n_conv)
        padding = _listify(padding, n_conv)
        stride = _listify(stride, n_conv)
        bias = _listify(bias, n_conv)

        self.latent_dim = (
            latent_dim + sum(_listify(reg_categorical))+ reg_gaussian)
        self.manifest_dim = manifest_dim
        self.discriminator = DualDiscriminator(
            channels=channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias, manifest_dim=manifest_dim,
            latent_dim=latent_dim, reg_categorical=reg_categorical,
            reg_gaussian=reg_gaussian)
        self.encoder = RegularisedEncoder(
            channels=channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias, manifest_dim=manifest_dim,
            hidden_dim=latent_dim, latent_noise_dim=latent_dim,
            reg_categorical=reg_categorical, reg_gaussian=reg_gaussian)
        self.generator = DCTranspose(
            channels=channels[::-1], kernel_size=kernel_size[::-1],
            stride=stride[::-1], padding=padding[::-1], bias=bias[::-1],
            latent_dim=latent_dim, target_dim=manifest_dim)

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

    def load_state_dict(self, params_g, params_e, params_d):
        self.encoder.load_state_dict(params_e)
        self.generator.load_state_dict(params_g)
        self.discriminator.load_state_dict(params_d=params_d)


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
        Q distribution network for informational regularisation.
    """
    def __init__(self,
                 manifest_dim=28,
                 latent_dim=100,
                 channels=(1, 128, 256, 512, 1024),
                 kernel_size=4,
                 stride=2,
                 padding=(2, 2, 1, 1),
                 bias=False,
                 reg_categorical=(10,),
                 reg_gaussian=2):
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
        reg_categorical: tuple
            List of level counts for uniformly categorically distributed
            variables in the latent space. For instance, (10, 8) denotes one
            variable with 10 levels and another with 8 levels.
        reg_gaussian: int
            Number of normally distributed variables in the latent space.
        """
        super(DualDiscriminator, self).__init__()
        self.x_discriminator = DCNetwork(
            channels=channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias, in_dim=manifest_dim,
            out_dim=latent_dim*2, final_act='leaky', embedded=(False, True))
        self.z_discriminator = DCNetwork(
            channels=(latent_dim,), fc=(latent_dim*2, latent_dim*2),
            kernel_size=1, stride=1, padding=0, bias=True, in_dim=1,
            out_dim=latent_dim*2, final_act='leaky', embedded=(False, True))
        self.zx_discriminator = DCNetwork(
            channels=(latent_dim*4,), fc=(latent_dim*4, latent_dim*4),
            kernel_size=1, stride=1, padding=0, bias=True, in_dim=1,
            out_dim=1, embedded=(True, False))
        self.regulariser = QStack(
            reg_categorical=reg_categorical, reg_gaussian=reg_gaussian,
            hidden_dim=latent_dim*2)

    def forward(self, z, x):
        z = self.z_discriminator(z)
        x = self.x_discriminator(x)
        zx = torch.cat([z, x], 1) + eps
        zx = self.zx_discriminator(zx)
        reg = self.regulariser(x)
        return zx, reg


class RegularisedEncoder(nn.Module):
    """
    An encoder network that encodes the manifest-space data into a latent-
    space representation that includes both regularised and noisy variables.

    Attributes
    ----------
    conv: DCNetwork
        The encoder's convolutional stack.
    code: ModuleDict
        The output layers of the encoder, which ensure that each distribution
        is appropriately processed.
    """
    def __init__(self,
                 channels=(1, 128, 256, 512, 1024),
                 kernel_size=4,
                 stride=2,
                 padding=(2, 2, 1, 1),
                 bias=False,
                 manifest_dim=28,
                 hidden_dim=256,
                 reg_categorical=(10,),
                 reg_gaussian=2,
                 latent_noise_dim=100):
        """Initialise an encoder network.

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
        manifest_dim: int
            Side length of the input image.
        hidden_dim: int
            Number of features in the final layer before the encoding layer.
        reg_categorical: tuple
            List of level counts for uniformly categorically distributed
            variables in the latent space. For instance, (10, 8) denotes one
            variable with 10 levels and another with 8 levels.
        reg_gaussian: int
            Number of regularised normally distributed variables in the latent
            space.
        latent_noise_dim: int
            Number of features in the latent space that are not regularised.
        """
        super(RegularisedEncoder, self).__init__()
        self.conv = DCNetwork(
            channels=channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias, in_dim=manifest_dim,
            out_dim=hidden_dim, final_act='leaky', embedded=(False, True))
        self.code = nn.ModuleDict()

        self.code['z'] = nn.Conv2d(
            in_channels=hidden_dim, out_channels=latent_noise_dim,
            kernel_size=1, stride=1, padding=0, bias=True)
        for i, levels in enumerate(reg_categorical):
            self.code['cat{}'.format(i)] = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dim, out_channels=levels,
                    kernel_size=1, stride=1, padding=0, bias=True),
                nn.Softmax()
            )
        if reg_gaussian > 0:
            self.code['gaussian'] = nn.Conv2d(
                in_channels=hidden_dim, out_channels=reg_gaussian,
                kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c = {}
        zc = self.conv(x)
        z = self.code['z'](zc)
        for i, levels in enumerate(reg_categorical):
            c['cat{}'.format(i)] = self.code['cat{}'.format(i)](zc)
        if reg_gaussian > 0:
            c['gaussian'] = self.code['gaussian'](zc)
        return z, c


class QStack(nn.Module):
    """Q distribution neural network for informational regularisation.

    Attributes
    ----------
    q_regularised: ModuleDict
        Dictionary of modules that yield distributional parameters for the
        compressible input c.
    reg_categorical: tuple
        List of level counts for uniformly categorically distributed variables
        in the latent space. For instance, (10, 8) denotes one variable with
        10 levels and another with 8 levels.
    reg_gaussian: int
        Number of regularised Gaussian variables in the latent space.
    """
    def __init__(self,
                 reg_categorical=(10,),
                 reg_gaussian=2,
                 hidden_dim=100):
        """Initialise a Q stack.

        Parameters
        ----------
        reg_categorical: tuple
            List of level counts for uniformly categorically distributed
            variables in the latent space. For instance, (10, 8) denotes one
            variable with 10 levels and another with 8 levels.
        reg_gaussian: int
            Number of regularised normally distributed variables in the latent
            space.
        hidden_dim: int
            Dimensionality of the hidden layer.
        """
        super(QStack, self).__init__()
        reg_categorical = list(reg_categorical)
        self.q_regularised = nn.ModuleDict()
        self.reg_gaussian = reg_gaussian
        self.reg_categorical = len(reg_categorical)
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
        for i, levels in enumerate(reg_categorical):
            self.q_regularised['cat{}'.format(i)] = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dim, out_channels=levels,
                    kernel_size=1, stride=1, padding=0, bias=True),
                nn.Softmax()
            )
        if reg_gaussian > 0:
            self.q_regularised['gaussian'] = nn.ModuleDict({
                'mean': nn.Conv2d(
                    in_channels=hidden_dim, out_channels=reg_gaussian,
                    kernel_size=1, stride=1, padding=0, bias=True),
                'logstd': nn.Conv2d(
                    in_channels=hidden_dim, out_channels=reg_gaussian,
                    kernel_size=1, stride=1, padding=0, bias=True)
            })

    def forward(self, x):
        c = {}
        x = self.q_input(x)
        for i in range(self.reg_categorical):
            c['cat{}'.format(i)] = self.q_regularised['cat{}'.format(i)](x)
        if self.reg_gaussian > 0:
            c['gaussian'] = {}
            c['gaussian']['mean'] = self.q_regularised['gaussian']['mean'](x)
            c['gaussian']['logstd'] = (
                self.q_regularised['gaussian']['logstd'](x))
        return c
