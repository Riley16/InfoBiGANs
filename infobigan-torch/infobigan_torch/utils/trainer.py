# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generic trainer for torch networks.
"""


class Trainer(object):
    """
    Generic trainer class.

    Attributes
    ----------
    loader: DataLoader
        DataLoader for the dataset to be used for training.
    model: Module
        Network to be trained.
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
    cuda: bool
        Indicates whether the model should be trained on CUDA.
    """
    def __init__(self,
                 loader,
                 model,
                 batch_size,
                 learning_rate,
                 max_epoch,
                 cuda=False):

        self.loader = loader
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.cuda = cuda

        if self.cuda and torch.cuda.is_available():
            self.model.cuda()

        self.n_features = loader.dataset.train_data[0].nelement()
        data_dim = list(loader.dataset.train_data[0].shape)
        while len(data_dim) < 3:
            data_dim = [1] + data_dim
        self.data_dim = data_dim

    def to_vectors(self, obs, n_features=None):
        """Convert the observations in a mini-batch into vectors."""
        n_features = n_features or self.n_features
        return obs.view(-1, n_features)

    def to_images(self, vectors, data_dim=None):
        """Convert a set of vectors into images."""
        data_dim = data_dim or self.data_dim
        return vectors.view([-1] + data_dim)

    def report(self, epoch, data, loss, name=''):
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
        print('Epoch [{}/{} ({:.0f}%)]\t{} Loss [{:.6f}]'.format(
              epoch + 1, self.max_epoch,
              100 * (epoch + 1) / self.max_epoch,
              name, loss.item() / len(data)))
