import numpy as np
import torch
from plot_utils import save_plot, set_plot_style
from q2_neural_network_numpy import read_data, normalize


def entry_point():
    """
    Ready? Go!
    The entry point of the program.
    """

    np.random.seed(0)
    X, y, df = read_data('data/ps5_data.csv')
    x, _, _ = normalize(X)

    n_inputs = x.shape[1]
    n_hidden = 3

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Construct the model.
    model = torch.nn.Sequential(
        torch.nn.Linear(n_inputs, n_hidden),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_hidden, 1),
    )

    loss_fn = torch.nn.MSELoss(reduction='sum')

    epochs = 30000
    learning_rate = 1e-3

    losses = np.empty(epochs)

    for t in range(epochs):
        # Forward pass.
        y_pred = model(x)

        # Compute loss.
        loss = loss_fn(y_pred, y)

        if t % 100 == 99:
            print(t, loss.item())

        losses[t] = loss.item()

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass.
        loss.backward()

        # Update the weights using gradient descent.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad




if __name__ == "__main__":
    set_plot_style()
    entry_point()
    print('We are done')
