"""
Compare values of loss functions between different models:
    numpy, pytorch
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from plot_utils import save_plot, set_plot_style
from q1_plot_data import TYPE1_EDGE_COLOR, TYPE2_EDGE_COLOR


def load_losses_from_cache(cache_dir):
    """
    Loads values of the loss function from a local files.

    Returns
    -------
    losses:
        See q2_variables.md.
    """

    return np.load(os.path.join(cache_dir, 'losses.npy'))


def plot_losses(q2_losses, q3_losses, skip_epochs, plot_dir, ylim=[0, 20]):
    """
    Plots the values of the loss function over iterations (epochs).
    The plot is saved to a file.

    Parameters
    ----------

    q2_losses, q3_losses: list of floats
        Values of the loss function at subsequent epoch for q2 and q3 models.

    skip_epochs: int
        Number of epochs skipped before storing the loss during model training.

    ylim: list
        Y axis limits: [min, max]

    plot_dir: str
        Directory for the output plot file.
    """
    fig, ax = plt.subplots()
    ax.plot(q2_losses, zorder=2, color=TYPE1_EDGE_COLOR, label='Numpy')
    ax.plot(q3_losses, zorder=2, color=TYPE2_EDGE_COLOR, label='Pytorch')
    ax.set_xlabel(f"Epoch (x{skip_epochs})")
    ax.set_ylabel('Loss')
    ax.set_ylim(ylim)
    ax.grid(zorder=1)
    ax.legend()
    fig.tight_layout(pad=0.20)
    save_plot(plt, file_name='losses_compared', subdir=plot_dir)


def entry_point():
    """
    Ready? Go!
    The entry point of the program.
    """
    q2_losses = load_losses_from_cache('model_cache/q2')
    q3_losses = load_losses_from_cache('model_cache/q3')

    plot_losses(
        q2_losses,
        q3_losses,
        skip_epochs=100,
        plot_dir='plots',
        ylim=[0, 20])


if __name__ == "__main__":
    set_plot_style()
    entry_point()
    print('We are done')
