# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
General utilities.
"""
import os
import glob
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


eps = 1e-15


def _listify(item, length=False):
    if length:
        if not (isinstance(item, tuple) or isinstance(item, list)):
            return [item] * length
        else:
            return list(item)
    else:
        if not (isinstance(item, tuple) or isinstance(item, list)):
            return [item]
        else:
            return list(item)


def thumb_grid(im_batch, grid_dim=(4, 4), im_dim=(6, 6),
               save=False, file='example.png', cuda=False):
    """Generate a grid of image thumbnails.

    Parameters
    ----------
    im_batch: Tensor
        Tensor of dimensionality (number of images) x (channels)
        x (height) x (width)
    grid_dim: tuple
        Dimensionality of the grid where the thumbnails should be plotted.
    im_dim: tuple
        Size of the image canvas.
    save: bool
        Indicates whether the thumbnails should be saved as a single image.
    file: str
        File where the image should be saved if `save` is true.
    """
    fig = plt.figure(1, im_dim)
    grid = ImageGrid(fig, 111, nrows_ncols=grid_dim, axes_pad=0.05)
    for i in range(im_batch.size(0)):
        if cuda:
            img = im_batch[i, :, :, :].detach().cpu().numpy().squeeze()
        else:
            img = im_batch[i, :, :, :].detach().numpy().squeeze()
        grid[i].imshow(img, cmap='bone')
        grid[i].axes.get_xaxis().set_visible(False)
        grid[i].axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig(file, bbox_inches='tight')


def animate_gif(out, src_fmt, duration=0.1, delete=False):
    """Animate a GIF of the training process.

    Parameters
    ----------
    out: str
        Path where the animated image should be saved.
    src_fmt: str
        Generic path to source images to be used in the animation. Any
        instances of the string `{epoch}` will be replaced by the wildcard
        (`*`) and results sorted.
    duration: float
        Duration of each frame, in seconds.
    delete: bool
        Indicates whether the source images used to compile the GIF animation
        should be deleted.
    """
    print('[Animating]')
    files = sorted(glob.glob(src_fmt.format(epoch='*'))) 
    with imageio.get_writer(out, mode='I', duration=duration) as writer:
        for file in files:
            img = imageio.imread(file)
            writer.append_data(img)
            if delete:
                os.remove(file)
    print('[Animation ready]')
