# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data loaders
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from os.path import dirname


def load_mnist(batch_size=100, train=True,
               renorm_mean=0.5, renorm_std=0.5):
    """Load the MNIST dataset into memory.

    Parameters
    ----------
    batch_size: int
        Number of observations in each mini-batch.
    train: bool
        Indicates whether the training set should be loaded.
    renorm_mean: float
        The renormalisation factor for the mean over all channels.
    renorm_std: float
        The renormalisation factor for the standard deviation over all
        channels. Renormalisation is performed as:

        renorm[channel] = (input[channel] - mean[channel]) / std[channel]

        Set `renorm_mean` to 0 and `renorm_std` to 1 to skip this
        transformation.

    Returns
    -------
    DataLoader
        DataLoader for the MNIST dataset.
    int
        The total number of mini-batches.

    `load_mnist` executes the following steps:
    (1) Download the MNIST data if they are not already present.
    (2) Renormalise the MNIST data. The data are initially stored with minimum
        intensity 0 and maximum intensity 1; the default renormalisation scales
        image intensities so that they fall in the range (-1, 1) instead.
    (3) Prepare a data loader for the provided batch size and determine the
        total number of batches.
    """
    normalise_data = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[renorm_mean],
                             std=[renorm_std])
    ])
    mnist_dir = '{}/mnist/'.format(dirname(__file__))

    mnist = datasets.MNIST(root=mnist_dir,
                           train=train,
                           transform=normalise_data,
                           download=True)
    loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)
    return loader
