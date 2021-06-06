"""
A neural network with a single layer that is trained to classify
the data. Plots the data, the loss function and the model predictions.

Based heavily on code provided in Andy Casey's lecture notes:
http://astrowizici.st/teaching/phs5000/13/


How to run
----------

See README.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from matplotlib.colors import LinearSegmentedColormap, ColorConverter
import colorsys
from plot_utils import save_plot, set_plot_style
from make_movie_from_images import make_movie_from_images

from q1_plot_data import plot_type, set_plot_limits, \
                         TYPE1_FACE_COLOR, TYPE2_FACE_COLOR, \
                         TYPE1_EDGE_COLOR, TYPE2_EDGE_COLOR


def read_data(path_to_data):
    """
    Reads the observation data.

    Parameters
    ----------
    path_to_data: str
        Path to CSV file containing the data.

    Returns
    --------

    (x, y, df): tuple
        df: Pandas' DataFrame
            The dataframe the CSV files was read into.

        x, y: See q2_variables.md.

    """

    df = pd.read_csv(path_to_data)
    x1 = df['x_1'].to_numpy().reshape((-1, 1))
    x2 = df['x_2'].to_numpy().reshape((-1, 1))
    x = np.hstack([x1, x2])
    y = df['classification'].to_numpy().reshape((-1, 1))

    return (x, y, df)


def normalize(x):
    """
    Transforms the input values so they have zero mean and
    standard deviation of 1.

    Parameters
    ----------
    x: See q2_variables.md.

    Returns
    -------
    numpy.ndarray 2D array
        Transformed values in the same format as the input.
    """

    x_mean, x_std = (np.mean(x, axis=0), np.std(x, axis=0))
    return (x - x_mean) / x_std, x_mean, x_std


def sigmoid(x):
    """
    Calculates the value of the sigmoid function.

    Parameters
    ----------
    x: float or numpy.ndarray
        Argument(s) of the sigmoid function.

    Returns
    -------
    float or numpy.ndarray
        Output of the sigmoid function.
    """

    return 1 / (1 + np.exp(-x))


def calculate_model_output(
        n_observations, inputs_with_bias, hidden_layer_weights,
        output_layer_weights):
    """
    Calculates the output value of the neural network given the
    input and weights.

    Parameters
    ----------

    See q2_variables.md.

    Returns
    -------

    (y_pred, hidden_layer_outputs): tuple
        See q2_variables.md.

    """

    hidden_layer_sums = inputs_with_bias @ hidden_layer_weights
    hidden_layer_outputs = sigmoid(hidden_layer_sums)

    output_layer_inputs = np.hstack([
        np.ones((n_observations, 1)),
        hidden_layer_outputs
    ])

    output = output_layer_inputs @ output_layer_weights

    return output, hidden_layer_outputs


def calculate_model_output_from_original_data(
    x, n_observations, hidden_layer_weights, output_layer_weights):
    """
    Same calculate_model_output `calculate_model_output` function
    but the original un-normalized numbers `x` are supplied instead
    """

    inputs_with_bias = make_input(x)

    y_pred, _ = calculate_model_output(
        inputs_with_bias=inputs_with_bias,
        n_observations=n_observations,
        hidden_layer_weights=hidden_layer_weights,
        output_layer_weights=output_layer_weights)

    return y_pred


def calculate_gradients(x, y, y_pred, n_hidden, hidden_layer_outputs,
                        output_layer_weights, gradients):
    """
    Calculate the gradients (dE/dw) for all the weights, which are
    derivatives of the loss function with respect to a gradients.

    The gradients are assigned to the elements in the `gradients` array.

    Parameters
    -----------

    See q2_variables.md.
    """

    s = (y_pred - y)
    n_inputs = x.shape[1]
    n_input_weights = (n_inputs + 1) * n_hidden

    # Input layer weights
    for i in range(n_input_weights):
        i_hidden = i % n_hidden  # Index of hidden neuron (not including bias)
        outer_weight = output_layer_weights[i_hidden + 1, 0]
        hidden_output = hidden_layer_outputs[:, [i_hidden]]
        weights = s * outer_weight * hidden_output * (1 - hidden_output)

        # Index of input neuron: -1 is bias, 0 is first input, 1 is second etc.
        i_input = int(math.floor((i - n_hidden) / n_hidden))

        if i_input >= 0:  # for non-bias node
            weights *= x[:, [i_input]]

        gradients[i] = np.sum(weights)

    # Hidden layer weights
    for i in range(len(output_layer_weights)):
        weight = s

        if i > 0:  # for non-bias node
            weight = s * hidden_layer_outputs[:, [i - 1]]

        gradients[n_input_weights + i] = np.sum(weight)


def update_weights(n_inputs, n_hidden, gradients, learning_rate, \
                   hidden_layer_weights, output_layer_weights):
    """
    Update the `hidden_layer_weights` and `output_layer_weights` weights,
    given the `gradients`.

    Parameters
    -----------

    learning_rate: int
        Learning rate, a small value like 0.001.

    See q2_variables.md.
    """

    n_input_weights = (n_inputs + 1) * n_hidden

    # Input layer weights
    scaled = learning_rate * gradients[:n_input_weights].reshape(
        n_inputs + 1, n_hidden)

    hidden_layer_weights -= scaled

    # Hidden layer weights
    scaled = learning_rate * gradients[n_input_weights:].reshape(n_hidden + 1, 1)
    output_layer_weights -= scaled


def loss_function(y, y_pred):
    """
    Calculates the value of the loss function.

    Parameters
    -----------

    See q2_variables.md.


    Returns
    -------

    float
        Value of the loss function.
    """

    # Note, there is no 0.5 factor here
    # to make it identical to torch.nn.MSELoss(reduction='sum')
    return np.sum((y_pred - y)**2)


def make_input(x):
    """
    Prepends column of 1's (the biases) to the model inputs.

    Parameters
    -----------

    x: See q2_variables.md.

    Returns
    -------

    inputs_with_bias:
        See q2_variables.md.
    """
    n_observations = x.shape[0]

    return np.hstack([
        np.ones((n_observations, 1)),
        x
    ])


def reshape_weights(hidden_weights, output_weights, n_inputs, n_hidden):
    """
    Given 1D arrays containing weights, return the weights in the specific
    shapes that are useful for faster calculations.

    Parameters
    ----------

    hidden_weights, output_weights: numpy.ndarray 1D array containing
        weights for connections coming into the hidden and output layers
        respsectively.

    n_inputs, n_hidden:
        See q2_variables.md.

    Returns
    -------

    (hidden_layer_weights, output_layer_weights): tuple
        See q2_variables.md.
    """

    hidden_layer_weights = hidden_weights.reshape(n_inputs + 1, n_hidden)
    output_layer_weights = output_weights.reshape(n_hidden + 1, 1)
    return hidden_layer_weights, output_layer_weights


def generate_weights(n_inputs, n_hidden):
    """
    Generate weights by darawing random numbers from a unit normal distribution.

    Parameters
    ----------

    See q2_variables.md.

    Returns
    -------

    (hidden_layer_weights, output_layer_weights): tuple
        See q2_variables.md.
    """

    hidden_weights = np.random.randn((n_inputs + 1) * n_hidden)
    output_weights = np.random.randn(n_hidden + 1)
    return reshape_weights(hidden_weights, output_weights, n_inputs, n_hidden)


def train_model(X, x, y, df, num_epochs, n_observations, n_hidden,
                inputs_with_bias,
                hidden_layer_weights, output_layer_weights, skip_epochs,
                predictions_plots_dir,
                predictions_plot_mesh_size):
    """
    Train the model by iterating `num_epochs` number of times and updating
    the model weights through backpropagation.

    Parameters
    ----------

    num_epochs: int
        Number of times to update the model weights thorugh backpropagation.

    skip_epochs: int
        Number of epochs to skip in the train loop before storing the
        value of the loss function in the returned loss array
        (so we don't output all losses, as the array will be too large).

    other parameters:
        See q2_variables.md.


    Returns
    -------

    losses:
        See q2_variables.md.
    """

    losses = np.empty(int(num_epochs / skip_epochs))
    n_out = 0
    n_inputs = x.shape[1]
    n_weights = (n_inputs + 1) * n_hidden + n_hidden + 1
    gradients = np.empty(n_weights)

    for epoch in range(num_epochs):
        y_pred, hidden_layer_outputs = calculate_model_output(
            inputs_with_bias=inputs_with_bias,
            n_observations=n_observations,
            hidden_layer_weights=hidden_layer_weights,
            output_layer_weights=output_layer_weights
        )

        calculate_gradients(
            x=x,
            y=y,
            y_pred=y_pred,
            n_hidden=n_hidden,
            hidden_layer_outputs=hidden_layer_outputs,
            output_layer_weights=output_layer_weights,
            gradients=gradients)

        update_weights(n_inputs=x.shape[1],
                       n_hidden=n_hidden,
                       gradients=gradients,
                       learning_rate=1e-3,
                       hidden_layer_weights=hidden_layer_weights,
                       output_layer_weights=output_layer_weights)

        if epoch % skip_epochs == 0:
            loss = loss_function(y, y_pred)
            print(epoch, loss)
            losses[n_out] = loss
            n_out += 1

            plot_predictions(
                X, df,
                mesh_size=predictions_plot_mesh_size,
                epoch=epoch,
                image_format='png',
                plot_dir=predictions_plots_dir,
                run_model_func=calculate_model_output_from_original_data,
                run_model_args={
                    "n_observations": predictions_plot_mesh_size,
                    "hidden_layer_weights": hidden_layer_weights,
                    "output_layer_weights": output_layer_weights
                },
                show_epoch=True
            )

    return losses


def plot_losses(losses, skip_epochs, plot_dir, ylim=[0, 20]):
    """
    Plots the values of the loss function over iterations (epochs).
    The plot is saved to a file.

    Parameters
    ----------

    skip_epochs: int
        Number of epochs skipped before storing the loss during model training.

    ylim: list
        Y axis limits: [min, max]

    plot_dir: str
        Directory for the output plot file.
    """
    fig, ax = plt.subplots()
    ax.plot(losses, zorder=2, color='#ff0021')
    ax.set_xlabel(f"Epoch (x{skip_epochs})")
    ax.set_ylabel('Loss')
    ax.set_ylim(ylim)
    ax.grid(zorder=1)
    fig.tight_layout(pad=0.20)
    save_plot(plt, file_name='loss', subdir=plot_dir)


def initialize_and_train_model(X, y, df, n_hidden, num_epochs, skip_epochs,
                               predictions_plots_dir,
                               predictions_plot_mesh_size):
    """
    Initializes the model weights and runs the model training given the
    input data.

    Parameters
    ----------
    X, y : 2D array
        Input data, not-normalized

    num_epochs: int
        Number of times to update the model weights thorugh backpropagation.

    skip_epochs: int
        Number of epochs to skip in the train loop before storing the
        value of the loss function in the returned loss array
        (so we don't output all losses, as the array will be too large).

    predictions_plots_dir: str
        Disrectory where prediction plots are saved.

    other parameters:
        See q2_variables.md.

    Returns
    -------

    (hidden_layer_weights, output_layer_weights, losses):
        See q2_variables.md.
    """

    x, _, _ = normalize(X)
    n_observations = x.shape[0]
    n_inputs = x.shape[1]
    inputs_with_bias = make_input(x)

    hidden_layer_weights, output_layer_weights = generate_weights(
        n_inputs, n_hidden
    )

    losses = train_model(
        X=X, x=x, y=y, df=df, num_epochs=num_epochs,
        n_observations=n_observations,
        n_hidden=n_hidden,
        inputs_with_bias=inputs_with_bias,
        hidden_layer_weights=hidden_layer_weights,
        output_layer_weights=output_layer_weights,
        skip_epochs=skip_epochs,
        predictions_plots_dir=predictions_plots_dir,
        predictions_plot_mesh_size=predictions_plot_mesh_size
    )

    return hidden_layer_weights, output_layer_weights, losses


def save_weights_to_cache(cache_dir, hidden_layer_weights,
                          output_layer_weights, losses):
    """
    Stores the arrays containing weights to files, so that we don't
    need to train the model if they exist.

    Parameters
    ----------

    cache_dir: str
        Path to the directory where the files with weights will be created.

    other parameters:
        See q2_variables.md.
    """

    os.makedirs(cache_dir, exist_ok=True)

    path = os.path.join(cache_dir, 'hidden_layer_weights')
    np.save(path, hidden_layer_weights)

    path = os.path.join(cache_dir, 'output_layer_weights')
    np.save(path, output_layer_weights)

    path = os.path.join(cache_dir, 'losses')
    np.save(path, losses)


def load_weights_from_cache(cache_dir):
    """
    Loads weights and values of the loss function for the model from local
    files, if they exist.

    Returns
    -------
    (hidden_layer_weights, output_layer_weights, losses): tuple
        See q2_variables.md.
    """
    try:
        hidden_layer_weights = np.load(os.path.join(cache_dir, 'hidden_layer_weights.npy'))
        output_layer_weights = np.load(os.path.join(cache_dir, 'output_layer_weights.npy'))
        losses = np.load(os.path.join(cache_dir, 'losses.npy'))
        return hidden_layer_weights, output_layer_weights, losses
    except IOError:
        return None, None, None


def train_model_or_get_weights_from_cache(
        X, y, df, n_hidden, num_epochs, cache_dir,
        skip_epochs, predictions_plots_dir, predictions_plot_mesh_size):
    """
    Initializes and trains the model. If model has already been trained
    and the weight files are stored in local files, then return the
    weights from cache instead.

    Parameters
    ----------

    predictions_plots_dir: str
        Directory where prediction plots are saved.

    other parameters:
        See q2_variables.md.

    Returns
    -------
    (hidden_layer_weights, output_layer_weights, losses): tuple
        See q2_variables.md.
    """

    this_dir = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.join(this_dir, cache_dir)

    hidden_layer_weights, output_layer_weights, losses = \
        load_weights_from_cache(dir)

    if hidden_layer_weights is not None:
        return hidden_layer_weights, output_layer_weights, losses

    hidden_layer_weights, output_layer_weights, losses = \
        initialize_and_train_model(
            X, y,
            df=df,
            n_hidden=n_hidden,
            num_epochs=num_epochs,
            skip_epochs=skip_epochs,
            predictions_plots_dir=predictions_plots_dir,
            predictions_plot_mesh_size=predictions_plot_mesh_size
        )

    save_weights_to_cache(cache_dir, hidden_layer_weights,
                          output_layer_weights, losses)

    return hidden_layer_weights, output_layer_weights, losses


def calc_prediction_mesh(X, mesh_size, padding,
                         run_model_func, run_model_args):
    """
    Calculates the data arrays (2d arrays called mesh) for predictions plot.

    Parameters
    ----------

    mesh_size: int
        The size of each of the two dimensions of the returned mesh arrays

    padding: int
        The proportion of the input range to be added to the left
        and right so that the inputs are not shown at the very
        edges of the plot.

    run_model_func: function
        Function that calculates model predictions.

    run_model_args: dict
        Arguments passed to run_model_func function.

    other parameters:
        See q2_variables.md.

    Returns
    -------
    (x1_mesh, x2_mesh, prediction_mesh): tuple of numpy.ndarray square 2D array
        x1_mesh, x2_mesh:
            The mash arrays containing the x1 and x2 values
            similar to the input data.

        prediction_mesh:
            Contains model prediction.
    """

    x, x_mean, x_std = normalize(X)
    x1_min, x2_min = x.min(axis=0)
    x1_max, x2_max = x.max(axis=0)

    # Add padding
    x1_range = x1_max - x1_min
    x1_min -= x1_range * padding
    x1_max += x1_range * padding
    x2_range = x2_max - x2_min
    x2_min -= x2_range * padding
    x2_max += x2_range * padding

    x1_grid = np.linspace(x1_min, x1_max, mesh_size)
    x2_grid = np.linspace(x2_min, x2_max, mesh_size)
    prediction_mesh = np.zeros([mesh_size, mesh_size])
    x1_grid = x1_grid.reshape((-1, 1))

    for i, x2 in enumerate(x2_grid):
        x2_single = np.array([x2] * len(x2_grid)).reshape((-1, 1))
        x_data = np.hstack([x1_grid, x2_single])
        y_pred = run_model_func(x=x_data, **run_model_args)
        prediction_mesh[i, :] = y_pred[:, 0]

    x1_grid_denormilized = x1_grid * x_std[0] + x_mean[0]
    x2_grid_denormilized = x2_grid * x_std[1] + x_mean[1]
    x1_mesh, x2_mesh = np.meshgrid(x1_grid_denormilized, x2_grid_denormilized)

    return x1_mesh, x2_mesh, prediction_mesh


def plot_observations(ax, df):
    """
    Adds observations to the plot.

    Parameters
    ----------

    ax: Matplotlib's axis

    df: Pandas' data frame.
        Contains input data.
    """
    edge = scale_lightness(TYPE1_EDGE_COLOR, 1.7)
    plot_type(ax, df, 0, marker='o', facecolor=TYPE1_FACE_COLOR, edgecolor=edge)

    edge = scale_lightness(TYPE2_EDGE_COLOR, 1.7)
    plot_type(ax, df, 1, marker='^', facecolor=TYPE2_FACE_COLOR, edgecolor=edge)


def scale_lightness(hex_color, scale_l):
    """
    Changes the brightness of the given color by the given factor `scale_l`.

    Parameters
    ----------

    hex_color: str
        Color in hex formar. Example: '#ff0000' for red.

    scale_l: float
        Scale factor by which to change the brightness. Value of 1 corresponds
        to no change.

    Source: https://stackoverflow.com/a/60562502/297131
    """

    rgb = ColorConverter.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def plot_predictions(X, df, mesh_size,
                     epoch, plot_dir, image_format,
                     run_model_func, run_model_args, show_epoch):
    """
    Plot the prediction of the model along with the input data.
    The plot is stored in a file.

    Parameters
    ----------

    df: Pandas' data frame.
        Contains input data.

    mesh_size: int
        Number of calculated predictions along each of the two plot axes.

    epoch: int
        The epoch index.

    plot_dir: str
        Dir where the plot is saved.

    image_format: str
        Format of the plot image: png, pdf, jpg.

    show_epoch: bool
        If True an epoch number is shown on the plot.

    run_model_func: function
        Function that calculates model predictions.

    run_model_args: dict
        Arguments passed to run_model_func function.

    other parameters:
        See q2_variables.md.

    """
    axis_padding = 0.05

    x, y, z = calc_prediction_mesh(
        X=X,
        mesh_size=mesh_size,
        padding=axis_padding,
        run_model_func=run_model_func,
        run_model_args=run_model_args
    )

    fig, ax = plt.subplots()

    colors = [
        scale_lightness(TYPE1_EDGE_COLOR, 1),
        scale_lightness(TYPE2_EDGE_COLOR, 1)
    ]

    cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=20)

    z = np.clip(z, 0, 1)  # consider values > 1 to be 1 and those < 0 to be 0
    pcm = ax.pcolormesh(x, y, z, cmap=cm, shading='gouraud', vmin=0, vmax=1)
    plot_observations(ax, df)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    set_plot_limits(ax, df, padding=axis_padding)

    fig.colorbar(pcm, ax=ax, label='Predicted classification')

    # Show epoch number
    if show_epoch:
        ax.text(
            0.02, 0.04,
            f'{epoch:05d}',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes,
            zorder=6,
            fontsize=11,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='0.7'))

    fig.tight_layout(pad=0.30)

    save_plot(
        plt,
        extensions=[image_format],
        subdir=plot_dir,
        file_name=f"predictions_epoch_{epoch:05d}"
    )

    fig.clear()
    plt.close()


def entry_point():
    """
    Ready? Go!
    The entry point of the program.
    """

    np.random.seed(0)
    X, y, df = read_data('data/ps5_data.csv')
    skip_epochs = 100
    num_epochs = 3000
    plot_frames_dir = 'plots/q2/movie_frames'
    plot_dir = 'plots/q2'
    predictions_plot_mesh_size = 300

    hidden_layer_weights, output_layer_weights, losses = \
        train_model_or_get_weights_from_cache(
            X=X, y=y, df=df,
            n_hidden=3, num_epochs=num_epochs, skip_epochs=skip_epochs,
            cache_dir='weights_cache',
            predictions_plots_dir=plot_frames_dir,
            predictions_plot_mesh_size=predictions_plot_mesh_size
        )

    plot_losses(losses, skip_epochs, plot_dir=plot_dir)

    plot_predictions(
        X, df,
        mesh_size=predictions_plot_mesh_size,
        epoch=int(num_epochs/skip_epochs),
        plot_dir=plot_dir,
        image_format='pdf',
        run_model_func=calculate_model_output_from_original_data,
        run_model_args={
            "n_observations": predictions_plot_mesh_size,
            "hidden_layer_weights": hidden_layer_weights,
            "output_layer_weights": output_layer_weights
        },
        show_epoch=False
    )

    make_movie_from_images(
        plot_dir=plot_frames_dir,
        movie_dir=plot_dir,
        movie_name='predictions.mp4',
        frame_rate=30
    )


if __name__ == "__main__":
    set_plot_style()
    entry_point()
    print('We are done')
