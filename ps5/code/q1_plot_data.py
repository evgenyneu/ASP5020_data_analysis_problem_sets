import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import save_plot, set_plot_style, MARKER_EDGE_WIDTH, MARKER_SIZE

TYPE1_FACE_COLOR = '#bcd5fdaa'
TYPE1_EDGE_COLOR = '#0060ff'
TYPE2_FACE_COLOR = '#febcc4aa'
TYPE2_EDGE_COLOR = '#ff0021'


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
        s=MARKER_SIZE,
        linewidth=MARKER_EDGE_WIDTH,
        zorder=5
    )


def set_plot_limits(ax, df, padding):
    """
    Sets the plot axis limit.

    Parameters
    ---------

    ax: Matplotib's plot axis.

    df: Panda's data frame containing the data for plotting.

    padding: float
        The proportion of the input range to be added to the left
        and right so that the inputs are not shown at the very
        edges of the plot.
    """
    x = df['x_1']
    y = df['x_2']

    x_range = x.max() - x.min()
    ax.set_xlim(x.min() - x_range * padding, x.max() + x_range * padding)

    y_range = y.max() - y.min()
    ax.set_ylim(y.min() - y_range * padding, y.max() + y_range * padding)


def plot_data(path_to_data, plot_dir):
    """
    Reads CSV file and plots the data.

    Parameters
    ---------

    path_to_data: string
        Path to the CSV file.

    plot_dir: string
        Path to plot directory.
    """

    df = pd.read_csv(path_to_data)
    fig, ax = plt.subplots()

    # Plot two types of data
    plot_type(ax, df, 0, marker='o', facecolor=TYPE1_FACE_COLOR, edgecolor=TYPE1_EDGE_COLOR)
    plot_type(ax, df, 1, marker='^', facecolor=TYPE2_FACE_COLOR, edgecolor=TYPE2_EDGE_COLOR)

    ax.set_xlabel(r'$x_1$')
    ax.grid(zorder=1)
    ax.set_ylabel(r'$x_2$')
    set_plot_limits(ax, df, padding=0.05)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.tight_layout(pad=0.20)
    save_plot(fig, subdir=plot_dir)


if __name__ == "__main__":
    set_plot_style()
    plot_data('data/ps5_data.csv', plot_dir='plots/q1')

    print('We are done')
