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
import os
import torch
from plot_utils import save_plot, set_plot_style

from q2_neural_network_numpy import (
    read_data, normalize, plot_losses, plot_predictions,
    make_movie_from_images
)


def calculate_model_output(x, model):
    with torch.no_grad():
        x = torch.tensor(x, dtype=torch.float32)
        return model(x)


def train_model(
    X, y, df, model, num_epochs, skip_epochs, learning_rate,
    plot_frames_dir, predictions_plot_mesh_size):

    x, _, _ = normalize(X)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    loss_fn = torch.nn.MSELoss(reduction='sum')
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
                X, df,
                mesh_size=predictions_plot_mesh_size,
                epoch=epoch,
                image_format='png',
                plot_dir=plot_frames_dir,
                run_model_func=calculate_model_output,
                run_model_args={
                    "model": model,
                },
                show_epoch=True
            )

    return losses


def save_model_to_cache(model, losses, cache_dir, mode_file_name):
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, mode_file_name)
    torch.save(model.state_dict(), model_path)
    path = os.path.join(cache_dir, 'losses')
    np.save(path, losses)


def load_model_from_cache(model, cache_dir, mode_file_name):
    model_path = os.path.join(cache_dir, "model.zip")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    losses = np.load(os.path.join(cache_dir, 'losses.npy'))
    return model, losses


def train_model_if_not_trained(
    X, y, df, model, num_epochs, skip_epochs, learning_rate,
    cache_dir,  plot_frames_dir, predictions_plot_mesh_size):

    this_dir = os.path.dirname(os.path.realpath(__file__))
    full_cache_dir = os.path.join(this_dir, cache_dir)
    mode_file_name = "model.zip"
    model_path = os.path.join(full_cache_dir, mode_file_name)

    if os.path.exists(model_path):
        model, losses = load_model_from_cache(model, cache_dir, mode_file_name)
    else:
        losses = train_model(
            X=X, y=y, df=df, model=model, num_epochs=num_epochs,
            skip_epochs=skip_epochs,
            learning_rate=learning_rate,
            plot_frames_dir=plot_frames_dir,
            predictions_plot_mesh_size=predictions_plot_mesh_size
        )

        save_model_to_cache(model, losses, cache_dir, mode_file_name)

    return losses


def entry_point():
    """
    Ready? Go!
    The entry point of the program.
    """

    np.random.seed(0)
    torch.manual_seed(0)
    plot_dir = 'plots/q3'
    plot_frames_dir = 'plots/q3/movie_frames'
    n_hidden = 3
    num_epochs = 30000
    skip_epochs = 100
    predictions_plot_mesh_size = 300
    learning_rate = 1e-3

    X, y, df = read_data('data/ps5_data.csv')
    n_inputs = X.shape[1]

    # Construct the model.
    model = torch.nn.Sequential(
        torch.nn.Linear(n_inputs, n_hidden),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_hidden, 1)
    )

    losses = train_model_if_not_trained(
        X=X, y=y, df=df, model=model, num_epochs=num_epochs,
        skip_epochs=skip_epochs,
        learning_rate=learning_rate,
        plot_frames_dir=plot_frames_dir,
        cache_dir='model_cache/q3',
        predictions_plot_mesh_size=predictions_plot_mesh_size
    )

    plot_losses(losses, skip_epochs, plot_dir=plot_dir)

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
