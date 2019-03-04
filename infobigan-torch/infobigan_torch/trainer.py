# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
InfoBiGAN trainer
~~~~~~~~~
Trainer class for the InfoBiGAN
"""
import torch
from torch import optim
from .conv import DCTranspose, DCNetwork
from .utils import eps, _listify
from pytorchure.train.trainer import Trainer
from pytorchure.utils import thumb_grid, animate_gif


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
    c = {}
    all_levels = (batch_size == 'levels')
    if all_levels:
        batch_size = levels[index]
    for i, n_levels in enumerate(levels):
        if all_levels:
            if index == i:
                c['cat{}'.format(i)] = torch.eye(n_levels)
            else:
                distr = torch.full((1, n_levels), 1/n_levels)
                sample = torch.multinomial(distr, 1).item()
                vectors = torch.zeros(1, n_levels)
                vectors[0, sample] = 1
                c['cat{}'.format(i)] = vectors.expand(batch_size, -1)
        else:
            distr = torch.full((1, n_levels), 1/n_levels)
            samples = torch.multinomial(distr, batch_size, replacement=True)
            vectors = torch.zeros(batch_size, n_levels)
            c['cat{}'.format(i)] = vectors.scatter_(1, samples.t(), 1)
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
    probe_dim = categorical_levels[index]
    c = categorical(
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
    probe_dim = len(gaussian_probe_default)
    z = gaussian(1, latent_noise).expand(probe_dim, -1)
    c = categorical(batch_size=1, levels=categorical_levels)
    for k, v in c.items():
        c[k] = c[k].expand(probe_dim, -1)
    c['gaussian'] = gaussian(probe_values, latent_gaussian, index)
    return c, z


def config_probe(latent_gaussian=2, categorical_levels=(10,),
                 latent_noise=100, probe_dim=16):
    """Generate a random probe in the latent space to qualitatively assess
    generator performance.

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
    probe_dim: int
        Total number of latent-space vectors to include in the probe.
    """
    z = gaussian(probe_dim, latent_noise)
    c = categorical(probe_dim, categorical_levels)
    c['gaussian'] = gaussian(probe_dim, latent_gaussian)
    return c, z
