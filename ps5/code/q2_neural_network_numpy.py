import numpy as np
from scipy.optimize import check_grad
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from plot_utils import save_plot, set_plot_style, MARKER_EDGE_WIDTH, MARKER_SIZE


def f(p, x, y, N):
    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13 = p

    # Hidden layer.
    hidden_layer_inputs = np.hstack([
        np.ones((N, 1)),
        x
    ])

    hidden_layer_weights = np.array([
        [w1,  w2,  w3],
        [w4,  w5,  w6],
        [w7,  w8,  w9]
    ])

    alpha_h = hidden_layer_inputs @ hidden_layer_weights
    beta_h = sigmoid(alpha_h)

    # Output layer.
    output_layer_inputs = np.hstack([
        np.ones((N, 1)),
        beta_h
    ])

    output_layer_weights = np.array([
        [w10, w11, w12, w13]
    ]).T

    alpha_o = output_layer_inputs @ output_layer_weights
    beta_o = alpha_o  # No activation function on output neuron
    y_pred = beta_o

    # Calculate our loss function: the average error in our predictions
    # compared to the target.
    # (This is also known as the mean squared error).
    return 0.5 * np.sum((y_pred - y)**2)


def g(p, x, y, N):
    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13 = p

    # Hidden layer.
    hidden_layer_inputs = np.hstack([
        np.ones((N, 1)),
        x
    ])

    hidden_layer_weights = np.array([
        [w1,  w2,  w3],
        [w4,  w5,  w6],
        [w7,  w8,  w9]
    ])

    alpha_h = hidden_layer_inputs @ hidden_layer_weights
    beta_h = sigmoid(alpha_h)

    # Output layer.
    output_layer_inputs = np.hstack([
        np.ones((N, 1)),
        beta_h
    ])

    output_layer_weights = np.array([
        [w10, w11, w12, w13]
    ]).T

    alpha_o = output_layer_inputs @ output_layer_weights

    # Calculate gradients
    s = (alpha_o - y)
    dE_dw13 = s * beta_h[:, [2]]
    dE_dw12 = s * beta_h[:, [1]]
    dE_dw11 = s * beta_h[:, [0]]
    dE_dw10 = s
    dE_dw9 = s * w13 * beta_h[:, [2]] * (1 - beta_h[:, [2]]) * x[:, [1]]
    dE_dw8 = s * w12 * beta_h[:, [1]] * (1 - beta_h[:, [1]]) * x[:, [1]]
    dE_dw7 = s * w11 * beta_h[:, [0]] * (1 - beta_h[:, [0]]) * x[:, [1]]
    dE_dw6 = s * w13 * beta_h[:, [2]] * (1 - beta_h[:, [2]]) * x[:, [0]]
    dE_dw5 = s * w12 * beta_h[:, [1]] * (1 - beta_h[:, [1]]) * x[:, [0]]
    dE_dw4 = s * w11 * beta_h[:, [0]] * (1 - beta_h[:, [0]]) * x[:, [0]]
    dE_dw3 = s * w13 * beta_h[:, [2]] * (1 - beta_h[:, [2]])
    dE_dw2 = s * w12 * beta_h[:, [1]] * (1 - beta_h[:, [1]])
    dE_dw1 = s * w11 * beta_h[:, [0]] * (1 - beta_h[:, [0]])

    return np.array([
        np.sum(dE_dw1),
        np.sum(dE_dw2),
        np.sum(dE_dw3),
        np.sum(dE_dw4),
        np.sum(dE_dw5),
        np.sum(dE_dw6),
        np.sum(dE_dw7),
        np.sum(dE_dw8),
        np.sum(dE_dw9),
        np.sum(dE_dw10),
        np.sum(dE_dw11),
        np.sum(dE_dw12),
        np.sum(dE_dw13)
    ])


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
    return 1 / (1 + np.exp(-x))


def run_model(N, hidden_layer_inputs, hidden_layer_weights, output_layer_weights):
    hidden_layer_sums = hidden_layer_inputs @ hidden_layer_weights
    hidden_layer_outputs = sigmoid(hidden_layer_sums)

    output_layer_inputs = np.hstack([
        np.ones((N, 1)),
        hidden_layer_outputs
    ])

    return output_layer_inputs @ output_layer_weights, hidden_layer_outputs


def calculate_gradients(x, y, N, y_pred, hidden_layer_outputs,
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
    hidden_layer_weights[0, 0] -= eta * gradients[0]  # w1
    hidden_layer_weights[0, 1] -= eta * gradients[1]  # w2
    hidden_layer_weights[0, 2] -= eta * gradients[2]  # w3
    hidden_layer_weights[1, 0] -= eta * gradients[3]  # w4
    hidden_layer_weights[1, 1] -= eta * gradients[4]  # w5
    hidden_layer_weights[1, 2] -= eta * gradients[5]  # w6
    hidden_layer_weights[2, 0] -= eta * gradients[6]  # w7
    hidden_layer_weights[2, 1] -= eta * gradients[7]  # w8
    hidden_layer_weights[2, 2] -= eta * gradients[8]  # w9
    output_layer_weights[0, 0] -= eta * gradients[9]  # w10
    output_layer_weights[1, 0] -= eta * gradients[10] # w11
    output_layer_weights[2, 0] -= eta * gradients[11] # w12
    output_layer_weights[3, 0] -= eta * gradients[12] # w13


def loss_function(y, y_pred):
    return 0.5 * np.sum((y_pred - y)**2)


def make_input(x, n_inputs):
    return np.hstack([
        np.ones((n_inputs, 1)),
        x
    ])


def generate_weights(n_inputs, n_hidden):
    weights = np.random.randn((n_inputs + 1) * n_hidden)
    hidden_layer_weights = weights.reshape(n_inputs + 1, n_hidden)
    output_layer_weights = np.random.randn(n_hidden + 1).reshape(n_hidden + 1, 1)
    return hidden_layer_weights, output_layer_weights


def run():
    """An entry point of the script"""

    np.random.seed(0)
    X, y = read_data()
    x = normalize(X)
    N = x.shape[0]  # Number of observations
    n_inputs = x.shape[1]
    n_hidden = 3  # The number of neurons in the hidden layer.
    assert check_grad(f, g, np.random.normal(size=13), x, y, N) < 1e-4
    hidden_layer_inputs = make_input(x, N)
    hidden_layer_weights, output_layer_weights = generate_weights(n_inputs, n_hidden)

    y_pred, _ = run_model(
        N, hidden_layer_inputs,
        hidden_layer_weights, output_layer_weights)

    print(f"Initial loss: {loss_function(y, y_pred):.0e}")

    # Back propagation
    # ---------

    num_epochs = 1000
    losses = np.empty(num_epochs)
    gradients = np.empty(13)

    for epoch in range(num_epochs):
        y_pred, hidden_layer_outputs = run_model(
            N, hidden_layer_inputs,
            hidden_layer_weights, output_layer_weights)

        calculate_gradients(x, y, N, y_pred, hidden_layer_outputs,
                            output_layer_weights, gradients)

        update_weights(gradients, 1e-3, hidden_layer_weights,
                       output_layer_weights)

        # Calculate our loss function
        loss = loss_function(y, y_pred)

        if not epoch % 100:
            print(epoch, loss)

        losses[epoch] = loss

    # Plot loss reducing with time
    # ---------

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


if __name__ == "__main__":
    run()
    print('We are done')
