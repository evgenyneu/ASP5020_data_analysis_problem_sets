import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import save_plot


def set_plot_style():
    """Set global style"""

    plt.rcParams['font.family'] = 'serif'

    TINY_SIZE = 15
    SMALL_SIZE = 18
    NORMAL_SIZE = 22
    LARGE_SIZE = 25

    # Title size
    plt.rcParams['axes.titlesize'] = LARGE_SIZE

    # Axes label size
    plt.rcParams['axes.labelsize'] = NORMAL_SIZE

    # Tick label size
    plt.rcParams['xtick.labelsize'] = TINY_SIZE
    plt.rcParams['ytick.labelsize'] = TINY_SIZE

    # Legend text size
    plt.rcParams['legend.fontsize'] = SMALL_SIZE

    plt.rcParams['font.size'] = NORMAL_SIZE
    plt.rcParams['legend.fontsize'] = NORMAL_SIZE

    # Legend location
    plt.rcParams["legend.loc"] = 'upper right'
    plt.rcParams["legend.framealpha"] = 0.9
    plt.rcParams["legend.edgecolor"] = '#000000'

    # Grid color
    plt.rcParams['grid.color'] = '#cccccc'

    # Define plot size
    plt.rcParams['figure.figsize'] = [12, 8]

    # Marker size
    plt.rcParams['lines.markersize'] = 25


def plot_type(ax, df, type, marker, facecolor, edgecolor):
    """
    Plots an observation type from the data.

    Parameters
    ---------

    ax:
        Matplotlib's axis

    df: Panda's data frame
        Data for plotting.

    type: int
        Type of data: 0 or 1

    marker, facecolor, edgecolor:
        Marker styles.
    """

    df_filtered = df.loc[df['classification'] == type]

    ax.scatter(
        df_filtered['x_1'],
        df_filtered['x_2'],
        marker=marker,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2,
        zorder=5
    )


def plot_data(path_to_data):
    """
    Reads CSV file and plots the data.

    Parameters
    ---------

    path_to_data: string
        Path to the CSV file.
    """

    df = pd.read_csv(path_to_data)
    fig, ax = plt.subplots()

    # Plot two types of data
    plot_type(ax, df, 0, marker='o', facecolor='#bcd5fdaa', edgecolor='#0060ff')
    plot_type(ax, df, 1, marker='^', facecolor='#febcc4aa', edgecolor='#ff0021')

    ax.set_xlabel(r'$x_1$')
    ax.grid(zorder=1)
    ax.set_ylabel(r'$x_2$')
    ax.set_title('Two types of observations')
    fig.tight_layout()
    save_plot(plt)


if __name__ == "__main__":
    set_plot_style()
    plot_data('ps5_data.csv')

    print('We are done')
