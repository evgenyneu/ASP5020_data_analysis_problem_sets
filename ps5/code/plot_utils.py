# Helper functions for dealing with plots

import os
import inspect
import numpy as np


def save_plot(plt, suffix=None, extensions=['pdf'], subdir='plots', dpi=300,
              silent=False):
    """
    Saves a plot to an image file. The name of the
    the image file is constructed from file name of the python script
    that called `plot_to_image` with an added `suffix`.

    The plot is saved to a

    Parameters
    -----------

    plt :
        Matplotlib's plot object

    suffix : str
        File name suffix for the output image file name.
        No suffix is used if None.

    extensions : list of str
        The output image file extensions which will be used to save the plot.

    subdir : str
        Directory where the plot will be placed.

    silent : bool
        If True will not print out the path the image is save to.
    """

    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    codefile = module.__file__

    this_dir = os.path.dirname(codefile)
    plot_dir = os.path.join(this_dir, subdir)
    os.makedirs(plot_dir, exist_ok=True)
    code_file_without_extension = os.path.basename(codefile).rsplit('.', 1)[0]

    if suffix is None:
        suffix = ''
    else:
        suffix = f'_{suffix}'

    for extension in extensions:
        filename = f'{code_file_without_extension}{suffix}.{extension}'
        figure_path = os.path.join(plot_dir, filename)
        plt.savefig(figure_path, dpi=dpi)
        printed_path = os.path.join(subdir, filename)
        print(f"Figure saved to {printed_path}")
