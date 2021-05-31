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


def run():
    """An entry point of the script"""

    np.random.seed(0)
    X, y = read_data()
    x = normalize(X)
    N = x.shape[0]  # Number of observations
    H = 3  # The number of neurons in the hidden layer.

    assert check_grad(f, g, np.random.normal(size=13), x, y, N) < 1e-4

    # Weights for the bias terms in input layer
    w1, w2, w3 = np.random.randn(H)

    # Weights for x1 to hidden layer neurons
    w4, w5, w6 = np.random.randn(H)

    # Weights for x2 to hidden layer neurons
    w7, w8, w9 = np.random.randn(H)

    # Weights for hidden layer outputs to output neuron
    w10, w11, w12, w13 = np.random.randn(H + 1)

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

    output_layer_weights = np.array([
        [w10, w11, w12, w13]
    ]).T

    output_layer_outputs, _ = run_model(
        N, hidden_layer_inputs,
        hidden_layer_weights, output_layer_weights)

    # Calculate our loss function: the total error in our predictions
    # compared to the target.
    loss = 0.5 * np.sum((output_layer_outputs - y)**2)

    print(f"Initial loss: {loss:.0e}")

    # Back propagation
    # ---------

    num_epochs = 1000
    losses = np.empty(num_epochs)

    eta = 1e-3
    for epoch in range(num_epochs):
        hidden_layer_weights = np.array([
            [w1,  w2,  w3],
            [w4,  w5,  w6],
            [w7,  w8,  w9]
        ])

        output_layer_weights = np.array([
            [w10, w11, w12, w13]
        ]).T

        y_pred, beta_h = run_model(
            N, hidden_layer_inputs,
            hidden_layer_weights, output_layer_weights)

        # Calculate our loss function: the average error in our predictions
        # compared to the target.
        # (This is also known as the mean squared error).
        loss = 0.5 * np.sum((y_pred - y)**2)
        if not epoch % 100:
            print(epoch, loss)

        losses[epoch] = loss

        # Calculate gradients
        s = (y_pred - y)
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

        # Now update the weights using stochastic gradient descent.
        w1 = w1 - eta * np.sum(dE_dw1)
        w2 = w2 - eta * np.sum(dE_dw2)
        w3 = w3 - eta * np.sum(dE_dw3)
        w4 = w4 - eta * np.sum(dE_dw4)
        w5 = w5 - eta * np.sum(dE_dw5)
        w6 = w6 - eta * np.sum(dE_dw6)
        w7 = w7 - eta * np.sum(dE_dw7)
        w8 = w8 - eta * np.sum(dE_dw8)
        w9 = w9 - eta * np.sum(dE_dw9)
        w10 = w10 - eta * np.sum(dE_dw10)
        w11 = w11 - eta * np.sum(dE_dw11)
        w12 = w12 - eta * np.sum(dE_dw12)
        w13 = w13 - eta * np.sum(dE_dw13)

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