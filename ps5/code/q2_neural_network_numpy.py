import numpy as np
import pandas as pd


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


def run():
    """An entry point of the script"""

    X, y = read_data()
    x = normalize(X)
    N = x.shape[0]  # Number of observations
    H = 3  # The number of neurons in the hidden layer.

    # Weights for the bias terms in input layer
    w1, w2, w3 = np.random.randn(H)

    # Weights for x1 to hidden layer neurons
    w4, w5, w6 = np.random.randn(H)

    # Weights for x2 to hidden layer neurons
    w7, w8, w9 = np.random.randn(H)

    # Weights for hidden layer outputs to output neuron
    w10, w11, w12, w13 = np.random.randn(H + 1)

    # Ok, let's code up our neural network!

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

    hidden_layer_sums = hidden_layer_inputs @ hidden_layer_weights
    hidden_layer_outputs = sigmoid(hidden_layer_sums)

    # Output layer.
    output_layer_inputs = np.hstack([
        np.ones((N, 1)),
        hidden_layer_outputs
    ])

    output_layer_weights = np.array([
        [w10, w11, w12, w13]
    ]).T

    output_layer_outputs = output_layer_inputs @ output_layer_weights

    # Calculate our loss function: the total error in our predictions
    # compared to the target.
    loss = 0.5 * np.sum((output_layer_outputs - y)**2)

    print(f"Initial loss: {loss:.0e}")


    # Back propagation
    # ---------

    num_epochs = 10000
    losses = np.empty(num_epochs)

    eta = 1e-3
    for epoch in range(num_epochs):
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
        loss = 0.5 * np.sum((y_pred - y)**2)
        if not epoch % 100:
            print(epoch, loss)

        losses[epoch] = loss

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


if __name__ == "__main__":
    run()
    print('We are done')
