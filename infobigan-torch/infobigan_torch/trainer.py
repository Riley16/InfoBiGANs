# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
InfoBiGAN trainer
~~~~~~~~~
Trainer class for the InfoBiGAN
"""
import torch
from torch import nn, optim
from .conv import DCTranspose, DCNetwork
from .utils.utils import thumb_grid, animate_gif
from .utils.trainer import Trainer


gaussian_probe_default = (-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)


def gaussian(batch_size, latent_dim=100, index=None):
    """
    Generate a 1D vector sampled from a Gaussian distribution.

    Parameters
    ----------
    batch_size: int or list
        Number of vectors to generate. If this is a list, it instead indicates
        the values to sample the distribution of the variable specified by
        `index`.
    latent_dim: int
        Number of latent features per vector.
    index: int
        Only used if `batch_size` is a list. Indicates the index of the
        variable that should be sampled at the specified values while the
        remaining variables are clamped.
    """
    if index is not None:
        values = batch_size
        batch_size = len(batch_size)
        samples = torch.randn(1, latent_dim).repeat(batch_size, 1)
        samples[:, index] = torch.Tensor(values)
        return samples
    else:
        return torch.randn(batch_size, latent_dim)


def categorical(batch_size, levels=(10,), index=None):
    """
    Generate a set of one-hot vectors sampled from a uniform categorical
    distribution.

    Parameters
    ----------
    batch_size: int or `levels`
        Number of vectors to generate. If this is `levels`, then one vector
        will instead be generated at each level of the variable specified by
        `index`.
    levels: list
        List of level counts for the uniformly categorically distributed
        variables to sample. For instance, (10, 8) denotes one variable with
        10 levels and another with 8 levels.
    index: int
        Only used if `batch_size` is set to `levels`. Indicates the index of
        the categorical variable that should be fully sampled while the others
        are clamped.
    """
    c = [None] * len(levels)
    all_levels = (batch_size == 'levels')
    if all_levels:
        batch_size = levels[index]
    for i, n_levels in enumerate(levels):
        if all_levels:
            if index == i:
                c[i] = torch.eye(n_levels)
            else:
                distr = torch.full((1, n_levels), 1/n_levels)
                sample = torch.multinomial(distr, 1).item()
                vectors = torch.zeros(1, n_levels)
                vectors[0, sample] = 1
                c[i] = vectors.expand(batch_size, -1)
        else:
            distr = torch.full((1, n_levels), 1/n_levels)
            samples = torch.multinomial(distr, batch_size, replacement=True)
            vectors = torch.zeros(batch_size, n_levels)
            c[i] = vectors.scatter_(1, samples.t(), 1)
    return c


def config_probe_categorical(categorical_levels=(10,), index=0,
                             latent_noise=100, latent_gaussian=2):
    """
    Generate a probe for a categorical variable of choice. The probe samples
    all possible levels of the selected variable while clamping all remaining
    variables.

    Parameters
    ----------
    categorical_levels: list
        List of level counts for the uniformly categorically distributed
        variables to sample. For instance, (10, 8) denotes one variable with
        10 levels and another with 8 levels.
    index: int
        Index of the variable that should be probed at all levels while the
        remaining variables are clamped.
    latent_noise: int
        Dimensionality of the latent-space noise variables.
    latent_gaussian: int
        Dimensionality of the regularised latent-space Gaussian variables.
    """
    c = {}
    probe_dim = categorical_levels[index]
    c['categorical'] = categorical(
        batch_size='levels', levels=categorical_levels, index=index)
    z = gaussian(1, latent_noise).expand(probe_dim, -1)
    c['gaussian'] = gaussian(1, latent_gaussian).expand(probe_dim, -1)
    return c, z


def config_probe_gaussian(latent_gaussian=2, latent_noise=100,
                          categorical_levels=(10,), index=0,
                          probe_values=gaussian_probe_default):
    """Generate a probe for the Gaussian variable of choice. The probe
    samples a range of values of the selected variable while clamping all
    remaining variables.

    Parameters
    ----------
    latent_gaussian: int
        Dimensionality of the regularised latent-space Gaussian variables.
    latent_noise: int
        Dimensionality of the latent-space noise variables.
    categorical_levels: list
        List of level counts for the uniformly categorically distributed
        variables to sample. For instance, (10, 8) denotes one variable with
        10 levels and another with 8 levels.
    index: int
        Index of the variable that should be probed across a range of values
        while the remaining variables are clamped.
    probe_values: list
        Specifies the values of the given variable to be probed.
    """
    c = {}
    probe_dim = len(gaussian_probe_default)
    z = gaussian(1, latent_noise).expand(probe_dim, -1)
    c['categorical'] = categorical(batch_size=1, levels=categorical_levels)
    c['categorical'] = [i.expand(probe_dim, -1) for i in c['categorical']]
    c['gaussian'] = gaussian(probe_values, latent_gaussian, index)
    return c, z


def config_sample(latent_gaussian=2, categorical_levels=(10,),
                  latent_noise=100, dim=16):
    """Generate a random sample in the latent space.

    Parameters
    ----------
    latent_gaussian: int
        Dimensionality of the regularised latent-space Gaussian variables.
    categorical_levels: list
        List of level counts for the uniformly categorically distributed
        variables to sample. For instance, (10, 8) denotes one variable with
        10 levels and another with 8 levels.
    latent_noise: int
        Dimensionality of the latent-space noise variables.
    dim: int
        Total number of latent-space vectors to include in the sample.
    """
    c = {}
    z = gaussian(dim, latent_noise)
    c['categorical'] = categorical(dim, categorical_levels)
    c['gaussian'] = gaussian(dim, latent_gaussian)
    return c, z


def config_infobigan_loss(batch_size):
    """
    Configure the loss functions for the InfoBiGAN.

    THIS FUNCTION IS NOT EVEN CLOSE TO CORRECT. It's here as a temporary
    placeholder until we get the correct loss functions.

    Parameters
    ----------
    batch_size: int
        Number of observations per batch.

    Returns
    -------
    BCELoss
        The binary cross-entropy loss function.
    Tensor
        Target indicating that the disciminator predicts the input (latent,
        manifest) pair was processed by the generator.
        This is the discriminator's target for generator-processed pairs and
        the generator-encoder's target for encoder-processed pairs.
    Tensor
        Target indicating that the disciminator predicts the input (latent,
        manifest) pair was processed by the encoder.
        This is the discriminator's target for encoder-processed pairs and
        the generator-encoder's target for generator-processed pairs.
    """
    loss = nn.BCELoss()
    generator_target = torch.ones(batch_size, 1, 1, 1)
    encoder_target = torch.zeros(batch_size, 1, 1, 1)
    return loss, generator_target, encoder_target


class InfoBiGANTrainer(Trainer):
    """
    Trainer class for an information maximising adversarially learned
    inference network (InfoBiGAN).

    Attributes
    ----------
    loader: DataLoader
        DataLoader for the dataset to be used for training.
    model: Module
        InfoBiGAN to be trained.
    batch_size: int
        Number of observations per mini-batch.
    learning_rate: float
        Optimiser learning rate.
    max_epoch: int
        Number of epochs of training.
    n_features: int
        Total number of features in each observation of the input dataset.
    data_dim: int
        Dimensionality of the input dataset. Extended to three dimensions
        (width, height, channels) to enable conversion to image.
    """
    def __init__(self,
                 loader,
                 model,
                 batch_size=100,
                 learning_rate=0.0002,
                 max_epoch=20):
        super(InfoBiGANTrainer, self).__init__(loader, model, batch_size,
                                               learning_rate, max_epoch)

        self.optimiser_d = optim.Adam(self.model.discriminator.parameters(),
                                      lr=self.learning_rate/2)
        self.optimiser_g = optim.Adam(self.model.generator.parameters(),
                                      lr=self.learning_rate)
        self.optimiser_e = optim.Adam(self.model.encoder.parameters(),
                                      lr=self.learning_rate)
        (self.loss,
         self.target_g,
         self.target_e) = config_infobigan_loss(self.batch_size)

    def train(self, log_progress=True, save_images=True,
              log_interval=100, img_prefix='infobigan'):
        """Train the InfoBiGAN.

        Parameters
        ----------
        . . .
        """
        self.model.train()
        c_probe, z_probe = config_sample(
            latent_gaussian=self.model.reg_gaussian,
            categorical_levels=self.model.reg_categorical,
            latent_noise=self.model.latent_noise,
            dim=16)
        if save_images:
            save = -1
            image_inst = '{}'.format(img_prefix) + '_{epoch:03d}.png'
            image_inst_f = '{}'.format(img_prefix) + '_{epoch}.png'
            image_out = '{}.gif'.format(img_prefix)
            self._save_images(c_probe, z_probe, save, image_inst)

        for epoch in range(self.max_epoch):
            loss_d_epoch = 0
            loss_g_epoch = 0
            loss_e_epoch = 0
            self.make_smooth_targets(epoch + 1)
            for i, (x, _) in enumerate(self.loader):
                c, z =  config_sample(
                    latent_gaussian=self.model.reg_gaussian,
                    categorical_levels=self.model.reg_categorical,
                    latent_noise=self.model.latent_noise,
                    dim=self.batch_size)

                x_hat = self.model.generator((c, z)).detach()
                c_hat, z_hat = self._detached(*self.model.encoder(x))
                error_d, _, _ = self.train_discriminator(
                    *self._generator_encoder_data(c, z, x,
                                                  c_hat, z_hat, x_hat))

                x_hat = self.model.generator((c, z))
                c_hat, z_hat = self.model.encoder(x)
                error_g, error_e = self.train_generator_encoder(
                    *self._generator_encoder_data(c, z, x,
                                                  c_hat, z_hat, x_hat))
                loss_d_epoch += error_d
                loss_g_epoch += error_g
                loss_e_epoch += error_e

                if log_progress and (i % log_interval == 0):
                    self.batch_report(i, epoch, error_d, 'Discriminator')
                    self.batch_report(i, epoch, error_g, 'Generator')
                    self.batch_report(i, epoch, error_e, 'Encoder')
                if save_images and (i % log_interval == 0):
                    save += 1
                    self._save_images(c_probe, z_probe, save, image_inst)

    def train_discriminator(self, generator_data, encoder_data):
        """Evaluate the error of the InfoBiGAN's discriminator network for a
        single mini-batch of generator- and encoder-processed data.

        Parameters
        ----------
        generator_data: Tensor
            Mini-batch of observations sampled from the generator.
        encoder_data: Tensor
            Mini-batch of observations sampled from the encoder.
        """
        self.optimiser_d.zero_grad()

        prediction_g, q_g = self.model.discriminator(*generator_data)
        error_g = self.loss(prediction_g, self.target_g)
        error_g.backward()

        prediction_e, q_e = self.model.discriminator(*encoder_data)
        error_e = self.loss(prediction_e, self.target_e)
        error_e.backward()

        self.optimiser_d.step()
        return error_g + error_e, prediction_g, prediction_e

    def train_generator_encoder(self, generator_data, encoder_data):
        """Evaluate the error of the InfoBiGAN's generator and encoder
        networks for a single mini-batch of fabricated data.

        Parameters
        ----------
        generator_data: Tensor
            Mini-batch of observations sampled from the generator.
        encoder_data: Tensor
            Mini-batch of observations sampled from the encoder.
        """
        self.optimiser_g.zero_grad()
        prediction_g, q_g = self.model.discriminator(*generator_data)
        error_g = self.loss(prediction_g, self.target_e)
        error_g.backward()
        self.optimiser_g.step()

        self.optimiser_e.zero_grad()
        prediction_e, q_e = self.model.discriminator(*encoder_data)
        error_e = self.loss(prediction_e, self.target_g)
        error_e.backward()
        self.optimiser_e.step()

        return error_g, error_e

    def make_smooth_targets(self, epoch, max_sigma=0.2):
        """Some crude and poorly researched label smoothing, because the
        discriminator is too good at this game. This smooths real image
        labels, with some probability of a label flip. Smoothing and flip
        probability decrease as training progresses.
        """
        prob_label_flip = 1 / (2 * epoch)
        label_flip = (torch.rand(self.batch_size, 1, 1, 1).abs_()
                      < prob_label_flip).float()
        noise_sigma = -max_sigma/self.max_epoch * epoch + max_sigma
        noise = (noise_sigma * torch.randn(self.batch_size, 1, 1, 1).abs_()
                 * (1 - label_flip))
        self.target_gd = torch.ones(self.batch_size, 1, 1, 1)
        self.target_ed = label_flip + noise

    def _detached(self, c, z):
        return {
            'categorical': [i.detach() for i in c['categorical']],
            'gaussian': c['gaussian'].detach()
        }, z.detach()

    def _generator_encoder_data(self, c, z, x, c_hat, z_hat, x_hat):
        generator_data = (
            (c, z),
            x_hat
        )
        encoder_data = (
            (c_hat, z_hat),
            x
        )
        return generator_data, encoder_data

    def batch_report(self, batch, epoch, loss, name=''):
        """Print a report on the current progress of training.

        Parameters
        ----------
        epoch: int
            Current epoch.
        data: Tensor
            Tensor containing all data for the current mini-batch.
        loss: Tensor
            Output of the loss function.
        name: str
            Name of the loss function (if there is more than one for the
            current network).
        """
        print('Batch [{}/{} ({:.0f}%)]\tEpoch [{}/{} ({:.0f}%)]\t'
              '{} Loss [{:.6f}]'.format(
                  batch + 1, len(self.loader),
                  100 * (batch + 1) / len(self.loader),
                  epoch + 1, self.max_epoch,
                  100 * (epoch + 1) / self.max_epoch,
                  name, loss.item()))

    def _save_images(self, c_probe, z_probe, save, image_fmt):
        """Save thumbnails of images generated during training."""
        probe_gen = self.model.generator((c_probe, z_probe))
        thumb_grid(probe_gen, save=True,
                   file=image_fmt.format(epoch=save + 1))
