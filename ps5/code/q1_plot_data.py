import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import save_plot, set_plot_style, MARKER_EDGE_WIDTH, MARKER_SIZE


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
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.tight_layout(pad=0.20)
    save_plot(fig)


if __name__ == "__main__":
    set_plot_style()
    plot_data('data/ps5_data.csv')

    print('We are done')
