import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from plot_utils import save_plot, set_plot_style, MARKER_EDGE_WIDTH, MARKER_SIZE


def read_data():
    df = pd.read_csv('data/ps5_data.csv')
    x1 = df['x_1'].to_numpy().reshape((-1, 1))
    x2 = df['x_2'].to_numpy().reshape((-1, 1))
    x = np.hstack([x1, x2])
    y = df['classification'].to_numpy().reshape((-1, 1))

    return (x, y)


def normalize(x):
    x_mean, x_std = (np.mean(x, axis=0), np.std(x, axis=0))
    return (x - x_mean) / x_std


def sigmoid(x):
    """
    Returns the value of the sigmoid function.
    """

    return 1 / (1 + np.exp(-x))


def run_model(n_observations, hidden_layer_inputs, hidden_layer_weights, \
              output_layer_weights):

    hidden_layer_sums = hidden_layer_inputs @ hidden_layer_weights
    hidden_layer_outputs = sigmoid(hidden_layer_sums)

    output_layer_inputs = np.hstack([
        np.ones((n_observations, 1)),
        hidden_layer_outputs
    ])

    return output_layer_inputs @ output_layer_weights, hidden_layer_outputs


def calculate_gradients(x, y, y_pred, hidden_layer_outputs,
                        output_layer_weights, gradients):
    w11 = output_layer_weights[1, 0]
    w12 = output_layer_weights[2, 0]
    w13 = output_layer_weights[3, 0]
    beta_h = hidden_layer_outputs

    s = (y_pred - y)
    gradients[12] = np.sum(s * beta_h[:, [2]])
    gradients[11] = np.sum(s * beta_h[:, [1]])
    gradients[10] = np.sum(s * beta_h[:, [0]])
    gradients[9] = np.sum(s)
    gradients[8] = np.sum(s * w13 * beta_h[:, [2]] * (1 - beta_h[:, [2]]) * x[:, [1]])
    gradients[7] = np.sum(s * w12 * beta_h[:, [1]] * (1 - beta_h[:, [1]]) * x[:, [1]])
    gradients[6] = np.sum(s * w11 * beta_h[:, [0]] * (1 - beta_h[:, [0]]) * x[:, [1]])
    gradients[5] = np.sum(s * w13 * beta_h[:, [2]] * (1 - beta_h[:, [2]]) * x[:, [0]])
    gradients[4] = np.sum(s * w12 * beta_h[:, [1]] * (1 - beta_h[:, [1]]) * x[:, [0]])
    gradients[3] = np.sum(s * w11 * beta_h[:, [0]] * (1 - beta_h[:, [0]]) * x[:, [0]])
    gradients[2] = np.sum(s * w13 * beta_h[:, [2]] * (1 - beta_h[:, [2]]))
    gradients[1] = np.sum(s * w12 * beta_h[:, [1]] * (1 - beta_h[:, [1]]))
    gradients[0] = np.sum(s * w11 * beta_h[:, [0]] * (1 - beta_h[:, [0]]))


def update_weights(gradients, eta, hidden_layer_weights, output_layer_weights):
    hidden_layer_weights[0, 0] -= eta * gradients[0]   # w1
    hidden_layer_weights[0, 1] -= eta * gradients[1]   # w2
    hidden_layer_weights[0, 2] -= eta * gradients[2]   # w3
    hidden_layer_weights[1, 0] -= eta * gradients[3]   # w4
    hidden_layer_weights[1, 1] -= eta * gradients[4]   # w5
    hidden_layer_weights[1, 2] -= eta * gradients[5]   # w6
    hidden_layer_weights[2, 0] -= eta * gradients[6]   # w7
    hidden_layer_weights[2, 1] -= eta * gradients[7]   # w8
    hidden_layer_weights[2, 2] -= eta * gradients[8]   # w9
    output_layer_weights[0, 0] -= eta * gradients[9]   # w10
    output_layer_weights[1, 0] -= eta * gradients[10]  # w11
    output_layer_weights[2, 0] -= eta * gradients[11]  # w12
    output_layer_weights[3, 0] -= eta * gradients[12]  # w13


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
    # Make the following inner weights for the inner layer:
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


def train_model(x, y, num_epochs, n_observations, hidden_layer_inputs,
                hidden_layer_weights, output_layer_weights):

    losses = np.empty(num_epochs)
    gradients = np.empty(13)

    for epoch in range(num_epochs):
        y_pred, hidden_layer_outputs = run_model(
            n_observations, hidden_layer_inputs,
            hidden_layer_weights, output_layer_weights)

        calculate_gradients(x, y, y_pred, hidden_layer_outputs,
                            output_layer_weights, gradients)

        update_weights(gradients, 1e-3, hidden_layer_weights,
                       output_layer_weights)

        # Calculate our loss function
        loss = loss_function(y, y_pred)

        if not epoch % 100:
            print(epoch, loss)

        losses[epoch] = loss

    return losses


def plot_losses(losses):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(losses, zorder=2, color='#ff0021')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_ylim([0, 10])
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(zorder=1)
    fig.tight_layout(pad=0.20)
    save_plot(plt, suffix='01')


def initialize_and_train_model(X, y):
    x = normalize(X)
    n_observations = x.shape[0]  # Number of observations
    n_inputs = x.shape[1]
    n_hidden = 3  # The number of neurons in the hidden layer.
    hidden_layer_inputs = make_input(x)

    hidden_layer_weights, output_layer_weights = generate_weights(
        n_inputs, n_hidden
    )

    losses = train_model(
        x=x, y=y, num_epochs=1000, n_observations=n_observations,
        hidden_layer_inputs=hidden_layer_inputs,
        hidden_layer_weights=hidden_layer_weights,
        output_layer_weights=output_layer_weights
    )

    return hidden_layer_weights, output_layer_weights, losses


def entry_point():
    np.random.seed(0)
    x, y = read_data()

    hidden_layer_weights, output_layer_weights, losses = \
        initialize_and_train_model(x, y)

    plot_losses(losses)


if __name__ == "__main__":
    entry_point()
    print('We are done')
