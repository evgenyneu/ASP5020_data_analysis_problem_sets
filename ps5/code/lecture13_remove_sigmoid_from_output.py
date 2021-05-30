import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from plot_utils import save_plot, set_plot_style, \
    MARKER_FACE_COLOR, MARKER_EDGE_COLOR, MARKER_EDGE_WIDTH, \
    MARKER_SIZE, LINE_COLOR

set_plot_style()

# For reproducibility
np.random.seed(0)

# Load in the data.
X = np.array([
    [2104, 3],
    [1600, 3],
    [2400, 3],
    [1416, 2],
    [3000, 4],
    [1985, 4],
    [1534, 3],
    [1427, 3],
    [1380, 3],
    [1494, 3],
])

Y = np.array([
    399900,
    329900,
    369000,
    232000,
    539900,
    299900,
    314900,
    198999,
    212000,
    242500,
]).reshape((-1, 1))

# For good reasons that we will explain later, we need to "normalise"
# the data so that it has zero mean and unit variance.
# (Just like what we did when we covered dimensionality reduction!)
X_mean, X_std = (np.mean(X, axis=0), np.std(X, axis=0))
x = (X - X_mean) / X_std

Y_mean, Y_std = (np.mean(Y), np.std(Y))
y = (Y - Y_mean) / Y_std

# Let's define some variables for the number of inputs and outputs.
N, D_in = x.shape
N, D_out = y.shape
H = 5 # The number of neurons in the hidden layer.

# Let's define our activation function.
sigmoid = lambda x: 1/(1 + np.exp(-x))

# Let's initialise all the weights randomly.
# (This is terrible code. I am only naming variables like this
# so you can see exactly how things work. You should use matrix
# multiplication!)

# Weights for the bias terms in the hidden layer:
w1, w2, w3, w4, w5 = np.random.randn(H)
# Weights for x1 to all neurons.
w6, w7, w8, w9, w10 = np.random.randn(H)
# Weights for x2 to all neurons.
w11, w12, w13, w14, w15 = np.random.randn(H)
# Weights for hidden layer outputs to output neuron.
w16, w17, w18, w19, w20, w21 = np.random.randn(H + 1)

# Ok, let's code up our neural network!

# Hidden layer.
hidden_layer_inputs = np.hstack([
    x,
    np.ones((N, 1))
])
hidden_layer_weights = np.array([
    [ w1,  w2,  w3,  w4,  w5],
    [ w6,  w7,  w8,  w9, w10],
    [w11, w12, w13, w14, w15]
])

hidden_layer_sums = hidden_layer_inputs @ hidden_layer_weights
hidden_layer_outputs = sigmoid(hidden_layer_sums)

# Output layer.
output_layer_inputs = np.hstack([
    hidden_layer_outputs,
    np.ones((N, 1))
])

output_layer_weights = np.array([
    [w16, w17, w18, w19, w20, w21]
]).T

output_layer_sums = output_layer_inputs @ output_layer_weights
output_layer_outputs = output_layer_sums  # <----- Removed sigmoid

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
        [ w1,  w2,  w3,  w4,  w5],
        [ w6,  w7,  w8,  w9, w10],
        [w11, w12, w13, w14, w15]
    ])

    alpha_h = hidden_layer_inputs @ hidden_layer_weights
    beta_h = sigmoid(alpha_h)

    # Output layer.
    output_layer_inputs = np.hstack([
        np.ones((N, 1)),
        beta_h
    ])
    output_layer_weights = np.array([
        [w16, w17, w18, w19, w20, w21]
    ]).T


    alpha_o = output_layer_inputs @ output_layer_weights
    beta_o = alpha_o  # <----- Removed sigmoid

    y_pred = beta_o

    # Calculate our loss function: the average error in our predictions compared to the target.
    # (This is also known as the mean squared error).
    loss = 0.5 * np.sum((y_pred - y)**2)
    if not epoch % 100:
        print(epoch, loss)

    losses[epoch] = loss

    # Calculate gradients.
    s = (beta_o - y)  # <----- Removed derivative of sigmoid
    dE_dw21 = s * beta_h[:, [4]]
    dE_dw20 = s * beta_h[:, [3]]
    dE_dw19 = s * beta_h[:, [2]]
    dE_dw18 = s * beta_h[:, [1]]
    dE_dw17 = s * beta_h[:, [0]]
    dE_dw16 = s
    dE_dw15 = s * w21 * beta_h[:, [4]] * (1 - beta_h[:, [4]]) * x[:, [1]]
    dE_dw14 = s * w20 * beta_h[:, [3]] * (1 - beta_h[:, [3]]) * x[:, [1]]
    dE_dw13 = s * w19 * beta_h[:, [2]] * (1 - beta_h[:, [2]]) * x[:, [1]]
    dE_dw12 = s * w18 * beta_h[:, [1]] * (1 - beta_h[:, [1]]) * x[:, [1]]
    dE_dw11 = s * w17 * beta_h[:, [0]] * (1 - beta_h[:, [0]]) * x[:, [1]]
    dE_dw10 = s * w21 * beta_h[:, [4]] * (1 - beta_h[:, [4]]) * x[:, [0]]
    dE_dw9  = s * w20 * beta_h[:, [3]] * (1 - beta_h[:, [3]]) * x[:, [0]]
    dE_dw8  = s * w19 * beta_h[:, [2]] * (1 - beta_h[:, [2]]) * x[:, [0]]
    dE_dw7  = s * w18 * beta_h[:, [1]] * (1 - beta_h[:, [1]]) * x[:, [0]]
    dE_dw6  = s * w17 * beta_h[:, [0]] * (1 - beta_h[:, [0]]) * x[:, [0]]
    dE_dw5  = s * w21 * beta_h[:, [4]] * (1 - beta_h[:, [4]])
    dE_dw4  = s * w20 * beta_h[:, [3]] * (1 - beta_h[:, [3]])
    dE_dw3  = s * w19 * beta_h[:, [2]] * (1 - beta_h[:, [2]])
    dE_dw2  = s * w18 * beta_h[:, [1]] * (1 - beta_h[:, [1]])
    dE_dw1  = s * w17 * beta_h[:, [0]] * (1 - beta_h[:, [0]])

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
    w14 = w14 - eta * np.sum(dE_dw14)
    w15 = w15 - eta * np.sum(dE_dw15)
    w16 = w16 - eta * np.sum(dE_dw16)
    w17 = w17 - eta * np.sum(dE_dw17)
    w18 = w18 - eta * np.sum(dE_dw18)
    w19 = w19 - eta * np.sum(dE_dw19)
    w20 = w20 - eta * np.sum(dE_dw20)
    w21 = w21 - eta * np.sum(dE_dw21)


# Plot loss reducing with time
# ---------

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(losses, zorder=2, color='#ff0021')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.grid(zorder=1)
fig.tight_layout(pad=0.20)
save_plot(plt, suffix='01')


# Compare predictions with data
# ------

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    Y / 1000,
    (y_pred * Y_std + Y_mean) / 1000,
    s=MARKER_SIZE,
    facecolor='#febcc4aa',
    edgecolor='#ff0021',
    linewidth=MARKER_EDGE_WIDTH,
    zorder=3
)

lims = np.array([
    ax.get_xlim(),
    ax.get_ylim()
])

lims = (np.min(lims), np.max(lims))

ax.plot(
    lims,
    lims,
    lw=1,
    ls=":",
    c="#666666",
    zorder=2
)

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Actual price USD')
ax.set_ylabel('Predicted price USD')
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.grid(zorder=1)
fig.tight_layout(pad=0.20)
save_plot(plt, suffix='02')

print('We are done!')

