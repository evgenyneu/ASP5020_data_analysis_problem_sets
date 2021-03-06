# Helper functions for dealing with plots

import os
import inspect
import matplotlib.pyplot as plt

# Line styles
LINE_COLOR = '#0060ff'

# Default marker styles
MARKER_SIZE = 200
MARKER_FACE_COLOR = '#bcd5fdaa'
MARKER_EDGE_COLOR = '#0060ff'
MARKER_EDGE_WIDTH = 1.5


def set_plot_style():
    """Set global style"""

    plt.rcParams['font.family'] = 'serif'

    TINY_SIZE = 13
    SMALL_SIZE = 18
    NORMAL_SIZE = 20
    LARGE_SIZE = 23

    # Title size
    plt.rcParams['axes.titlesize'] = LARGE_SIZE

    # Axes label size
    plt.rcParams['axes.labelsize'] = SMALL_SIZE

    # Tick label size
    plt.rcParams['xtick.labelsize'] = TINY_SIZE
    plt.rcParams['ytick.labelsize'] = TINY_SIZE

    # Text size
    plt.rcParams['font.size'] = NORMAL_SIZE

    # Legend location
    plt.rcParams["legend.loc"] = 'upper right'
    plt.rcParams["legend.framealpha"] = 0.9
    plt.rcParams["legend.edgecolor"] = '#000000'

    # Legend text size
    plt.rcParams['legend.fontsize'] = SMALL_SIZE

    # Grid color
    plt.rcParams['grid.color'] = '#cccccc'

    # Define plot size
    plt.rcParams['figure.figsize'] = [9, 6]

    # Lines
    plt.rcParams['lines.color'] = 'red'
    plt.rcParams['lines.linewidth'] = 2

    # Grid
    plt.rcParams['grid.color'] = '#555555'
    plt.rcParams['grid.alpha'] = 0.2


def save_plot(plt, file_name=None, suffix=None,
              extensions=['pdf'], subdir='plots', dpi=300,
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

    file_name: str
        Base name (name without extension) for the plot file.
        If None, the name of Python script that called this function
        will be used.

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

    if file_name is None:
        file_name = os.path.basename(codefile).rsplit('.', 1)[0]

    if suffix is None:
        suffix = ''
    else:
        suffix = f'_{suffix}'

    for extension in extensions:
        filename = f'{file_name}{suffix}.{extension}'
        figure_path = os.path.join(plot_dir, filename)
        plt.savefig(figure_path, dpi=dpi)
        printed_path = os.path.join(subdir, filename)

        if not silent:
            print(f"Figure saved to {printed_path}")
