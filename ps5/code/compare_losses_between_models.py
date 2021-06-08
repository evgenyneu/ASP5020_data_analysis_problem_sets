"""
Compare values of loss functions between different models:
    numpy, pytorch
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from plot_utils import save_plot, set_plot_style
from q1_plot_data import TYPE1_EDGE_COLOR, TYPE2_EDGE_COLOR
from itertools import cycle


def load_losses_from_cache(cache_dir):
    """
    Loads values of the loss function from a local files.

    Returns
    -------
    losses:
        See q2_variables.md.
    """

    return np.load(os.path.join(cache_dir, 'losses.npy'))


def plot_losses(all_losses, labels, skip_epochs, plot_dir, ylim=[0, 20]):
    """
    Plots the values of the loss function over iterations (epochs).
    The plot is saved to a file.

    Parameters
    ----------

    all_losses: list of lost of floats
        List containing losses for different codes (python, pytorch, tensorflow)

    labels: list of str
        Names of the codes corresponding to items in `all_losses` parameter.

    skip_epochs: int
        Number of epochs skipped before storing the loss during model training.

    ylim: list
        Y axis limits: [min, max]

    plot_dir: str
        Directory for the output plot file.
    """
    fig, ax = plt.subplots()
    line_styles = ["-", "--", "-.", ":"]
    line_style_cycler = cycle(line_styles)
    colors = [TYPE1_EDGE_COLOR, TYPE2_EDGE_COLOR, '#44aa22']
    color_cycler = cycle(colors)

    for losses, label in zip(all_losses, labels):
        ax.plot(
            losses,
            zorder=2,
            label=label,
            linestyle=next(line_style_cycler),
            color=next(color_cycler)
        )

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

    dirs = ['q2', 'q3_pytorch', 'q3_tensorflow']
    dirs = [f"model_cache/{dir}" for dir in dirs]
    losses = [load_losses_from_cache(dir) for dir in dirs]

    plot_losses(
        losses,
        labels=['Numpy', 'Pytorch', 'Tensorflow'],
        skip_epochs=100,
        plot_dir='plots',
        ylim=[0, 20])


if __name__ == "__main__":
    set_plot_style()
    entry_point()
    print('We are done')
