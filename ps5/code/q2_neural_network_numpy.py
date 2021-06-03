import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from matplotlib.colors import LinearSegmentedColormap, Normalize, ColorConverter
import colorsys
from plot_utils import save_plot, set_plot_style

from q1_plot_data import plot_type, set_plot_limits, \
                         TYPE1_FACE_COLOR, TYPE2_FACE_COLOR, \
                         TYPE1_EDGE_COLOR, TYPE2_EDGE_COLOR


def read_data(path_to_data):
    df = pd.read_csv(path_to_data)
    x1 = df['x_1'].to_numpy().reshape((-1, 1))
    x2 = df['x_2'].to_numpy().reshape((-1, 1))
    x = np.hstack([x1, x2])
    y = df['classification'].to_numpy().reshape((-1, 1))

    return (x, y, df)


def normalize(x):
    x_mean, x_std = (np.mean(x, axis=0), np.std(x, axis=0))
    return (x - x_mean) / x_std, x_mean, x_std


def sigmoid(x):
    """
    Returns the value of the sigmoid function.
    """

    return 1 / (1 + np.exp(-x))


def calculate_model_output(
        n_observations, hidden_layer_inputs, hidden_layer_weights,
        output_layer_weights):

    hidden_layer_sums = hidden_layer_inputs @ hidden_layer_weights
    hidden_layer_outputs = sigmoid(hidden_layer_sums)

    output_layer_inputs = np.hstack([
        np.ones((n_observations, 1)),
        hidden_layer_outputs
    ])

    return output_layer_inputs @ output_layer_weights, hidden_layer_outputs


def calculate_gradients(x, y, y_pred, n_hidden, hidden_layer_outputs,
                        output_layer_weights, gradients):

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


def update_weights(n_inputs, n_hidden, gradients, eta, hidden_layer_weights, \
                   output_layer_weights):

    n_input_weights = (n_inputs + 1) * n_hidden

    # Input layer weights
    scaled = eta * gradients[:n_input_weights].reshape(n_inputs + 1, n_hidden)
    hidden_layer_weights -= scaled

    # Hidden layer weights
    scaled = eta * gradients[n_input_weights:].reshape(n_hidden + 1, 1)
    output_layer_weights -= scaled


def loss_function(y, y_pred):
    return 0.5 * np.sum((y_pred - y)**2)


def make_input(x):
    """
    Given the observations `x`, appends the biases and returns
    [
        [x1, x2, 1],
        [x1, x2, 1],
        [x1, x2, 1],
        [x1, x2, 1]
        ...
        ...
    ]
    """
    n_observations = x.shape[0]

    return np.hstack([
        np.ones((n_observations, 1)),
        x
    ])


def reshape_weights(hidden_weights, output_weights, n_inputs, n_hidden):
    # Make the following inner weights for the inner layer (for 3 hidden neurons):
    # [
    #     [w1,  w2,  w3],
    #     [w4,  w5,  w6],
    #     [w7,  w8,  w9]
    # ]
    hidden_layer_weights = hidden_weights.reshape(n_inputs + 1, n_hidden)

    # Outer layer weights:
    # [
    #     [w10],
    #     [w11],
    #     [w12],
    #     [w13]
    # ]
    output_layer_weights = output_weights.reshape(n_hidden + 1, 1)

    return hidden_layer_weights, output_layer_weights


def generate_weights(n_inputs, n_hidden):
    hidden_weights = np.random.randn((n_inputs + 1) * n_hidden)
    output_weights = np.random.randn(n_hidden + 1)
    return reshape_weights(hidden_weights, output_weights, n_inputs, n_hidden)


def train_model(x, y, num_epochs, n_observations, n_hidden,
                hidden_layer_inputs,
                hidden_layer_weights, output_layer_weights, skip_epochs):

    losses = np.empty(int(num_epochs / skip_epochs))
    n_out = 0
    n_inputs = x.shape[1]
    n_weights = (n_inputs + 1) * n_hidden + n_hidden + 1
    print(n_weights)
    gradients = np.empty(n_weights)

    for epoch in range(num_epochs):
        y_pred, hidden_layer_outputs = calculate_model_output(
            n_observations, hidden_layer_inputs,
            hidden_layer_weights, output_layer_weights)

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
                       eta=1e-3,
                       hidden_layer_weights=hidden_layer_weights,
                       output_layer_weights=output_layer_weights)

        # Calculate our loss function
        loss = loss_function(y, y_pred)

        if not epoch % skip_epochs:
            print(epoch, loss)
            losses[n_out] = loss
            n_out += 1

    return losses


def plot_losses(losses, skip_epochs):
    fig, ax = plt.subplots()
    ax.plot(losses, zorder=2, color='#ff0021')
    ax.set_xlabel(f"Epoch (x{skip_epochs})")
    ax.set_ylabel('Loss')
    ax.set_ylim([0, 10])
    ax.grid(zorder=1)
    fig.tight_layout(pad=0.20)
    save_plot(plt, suffix='01')


def initialize_and_train_model(X, y, n_hidden, num_epochs, skip_epochs):
    """
    Parameters
    ----------
    X, y : 2D array
        Input data, not-normalized

    n_hidden: int
        The number of neurons in the hidden layer.
    """

    x, _, _ = normalize(X)
    n_observations = x.shape[0]
    n_inputs = x.shape[1]
    hidden_layer_inputs = make_input(x)

    hidden_layer_weights, output_layer_weights = generate_weights(
        n_inputs, n_hidden
    )

    losses = train_model(
        x=x, y=y, num_epochs=num_epochs,
        n_observations=n_observations,
        n_hidden=n_hidden,
        hidden_layer_inputs=hidden_layer_inputs,
        hidden_layer_weights=hidden_layer_weights,
        output_layer_weights=output_layer_weights,
        skip_epochs=skip_epochs
    )

    return hidden_layer_weights, output_layer_weights, losses


def save_weights_to_cache(cache_dir, hidden_layer_weights,
                          output_layer_weights, losses):

    os.makedirs(cache_dir, exist_ok=True)

    path = os.path.join(cache_dir, 'hidden_layer_weights')
    np.save(path, hidden_layer_weights)

    path = os.path.join(cache_dir, 'output_layer_weights')
    np.save(path, output_layer_weights)

    path = os.path.join(cache_dir, 'losses')
    np.save(path, losses)


def load_weights_from_cache(cache_dir):
    try:
        hidden_layer_weights = np.load(os.path.join(cache_dir, 'hidden_layer_weights.npy'))
        output_layer_weights = np.load(os.path.join(cache_dir, 'output_layer_weights.npy'))
        losses = np.load(os.path.join(cache_dir, 'losses.npy'))
        return hidden_layer_weights, output_layer_weights, losses
    except IOError:
        return None, None, None


def train_model_or_get_weights_from_cache(
        x, y, n_hidden, num_epochs, cache_dir, skip_epochs):

    this_dir = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.join(this_dir, cache_dir)

    hidden_layer_weights, output_layer_weights, losses = \
        load_weights_from_cache(dir)

    if hidden_layer_weights is not None:
        return hidden_layer_weights, output_layer_weights, losses

    hidden_layer_weights, output_layer_weights, losses = \
        initialize_and_train_model(x, y, n_hidden=n_hidden,
                                   num_epochs=num_epochs,
                                   skip_epochs=skip_epochs)

    save_weights_to_cache(cache_dir, hidden_layer_weights,
                          output_layer_weights, losses)

    return hidden_layer_weights, output_layer_weights, losses


def calc_prediction_mesh(X, y, hidden_layer_weights, output_layer_weights, mesh_size, padding):
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
        hidden_layer_inputs = make_input(x_data)

        y_pred, _ = calculate_model_output(
            mesh_size, hidden_layer_inputs,
            hidden_layer_weights, output_layer_weights)

        prediction_mesh[i, :] = y_pred[:, 0]

    x1_grid_denormilized = x1_grid * x_std[0] + x_mean[0]
    x2_grid_denormilized = x2_grid * x_std[1] + x_mean[1]
    x1_mesh, x2_mesh = np.meshgrid(x1_grid_denormilized, x2_grid_denormilized)

    return x1_mesh, x2_mesh, prediction_mesh


def plot_observations(ax, df):
    edge = scale_lightness(TYPE1_EDGE_COLOR, 1.7)
    plot_type(ax, df, 0, marker='o', facecolor=TYPE1_FACE_COLOR, edgecolor=edge)

    edge = scale_lightness(TYPE2_EDGE_COLOR, 1.7)
    plot_type(ax, df, 1, marker='^', facecolor=TYPE2_FACE_COLOR, edgecolor=edge)


def scale_lightness(hex_color, scale_l):
    """https://stackoverflow.com/a/60562502/297131"""
    rgb = ColorConverter.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def plot_predictions(X, y, df, hidden_layer_weights, output_layer_weights):
    axis_padding = 0.05

    x, y, z = calc_prediction_mesh(
        X=X,
        y=y,
        hidden_layer_weights=hidden_layer_weights,
        output_layer_weights=output_layer_weights,
        mesh_size=300,
        padding=axis_padding
    )

    fig, ax = plt.subplots()

    colors = [
        scale_lightness(TYPE1_EDGE_COLOR, 1),
        scale_lightness(TYPE2_EDGE_COLOR, 1)
    ]

    cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=20)

    z = np.clip(z, 0, 1)  # consider values > 1 to be 1 and those < 0 to be 0
    pcm = ax.pcolormesh(x, y, z, cmap=cm, shading='gouraud')
    plot_observations(ax, df)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    set_plot_limits(ax, df, padding=axis_padding)
    fig.colorbar(pcm, ax=ax, label='Predicted classification')
    fig.tight_layout(pad=0.30)
    save_plot(plt, suffix='02')


def entry_point():
    np.random.seed(0)
    x, y, df = read_data('data/ps5_data.csv')
    skip_epochs = 100

    hidden_layer_weights, output_layer_weights, losses = \
        train_model_or_get_weights_from_cache(
            x=x, y=y, n_hidden=3, num_epochs=100000, skip_epochs=skip_epochs,
            cache_dir='weights_cache')

    plot_losses(losses, skip_epochs)
    plot_predictions(x, y, df, hidden_layer_weights, output_layer_weights)


if __name__ == "__main__":
    set_plot_style()
    entry_point()
    print('We are done')
