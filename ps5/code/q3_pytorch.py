"""
A neural network with a single layer that is trained to classify
the data using Pytorch library. Plots the data, the loss function
and the model predictions.

Based heavily on code provided in Andy Casey's lecture notes:
http://astrowizici.st/teaching/phs5000/13/


How to run
----------

See README.md
"""

import numpy as np
import torch
from plot_utils import save_plot, set_plot_style
from q2_neural_network_numpy import read_data, normalize, plot_losses


def entry_point():
    """
    Ready? Go!
    The entry point of the program.
    """

    np.random.seed(0)
    torch.manual_seed(0)
    plot_dir = 'plots/q3'

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
        torch.nn.Linear(n_hidden, 1)
    )

    loss_fn = torch.nn.MSELoss(reduction='sum')
    num_epochs = 3000
    skip_epochs = 100
    learning_rate = 1e-3
    losses = np.empty(int(num_epochs / skip_epochs))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    n_out = 0

    for epoch in range(num_epochs):
        y_pred = model(x)  # Calculate model predictions
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()  # Calculate the gradients
        optimizer.step()  # Update the weights

        if epoch % skip_epochs == 0:
            print(epoch, loss.item())
            losses[n_out] = loss.item()
            n_out += 1

            plot_predictions(
                X, y, df,
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

    plot_losses(losses, skip_epochs, plot_dir=plot_dir)


if __name__ == "__main__":
    set_plot_style()
    entry_point()
    print('We are done')
