"""
A neural network with a single layer that is trained to classify
the data using Tensorflow library. Plots the data, the loss function
and the model predictions.

Based on: https://github.com/IvanBongiorni/TensorFlow2.0_Notebooks

How to run
----------

See README.md
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from plot_utils import set_plot_style

from q2_neural_network_numpy import (
    read_data, normalize, plot_losses, plot_predictions,
    make_movie_from_images
)


def calculate_model_output(x, model):
    """
    Calculates the output of the model given the input data `x`.

    Parameters
    ----------

    x:
        See q2_variables.md.

    model: Pytorch model


    Returns
    --------

    float
    The output predicted value of the model.
    """

    return model(x, training=False)


def train_model(
    X, y, df, model, num_epochs, skip_epochs, learning_rate,
    plot_frames_dir, predictions_plot_mesh_size):
    """
    Train the model by iterating `num_epochs` number of times and updating
    the model weights through backpropagation.

    Parameters
    ----------

    model: Pytorch model

    num_epochs: int
        Number of times to update the model weights thorugh backpropagation.

    skip_epochs: int
        Number of epochs to skip in the train loop before storing the
        value of the loss function in the returned loss array
        (so we don't output all losses, as the array will be too large).

    plot_frames_dir: str
        Directory where the prediction plots will be created for different
        epochs. The images will be used to create a movie.

    other parameters:
        See q2_variables.md.


    Returns
    -------

    losses:
        See q2_variables.md.
    """

    x, _, _ = normalize(X)
    losses = np.empty(int(num_epochs / skip_epochs))
    n_out = 0
    loss_fn = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            y_pred = model(x)  # Calculate model predictions
            loss = loss_fn(y_pred, y)

        # Calculate the gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Update the weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch % skip_epochs == 0:
            print(epoch, loss.numpy())
            losses[n_out] = loss.numpy()
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


def save_model_to_cache(model, losses, cache_dir, model_file_name):
    """
    Stores the model parameters to files, so that we don't
    need to train the model if they exist.

    Parameters
    ----------

    model: Pytorch model

    cache_dir: str
        Path to the directory where the files with weights will be created.

    model_file_name: str
        Name of the file in which model parameters are stored.

    other parameters:
        See q2_variables.md.
    """

    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, model_file_name)
    model.save_weights(model_path)
    path = os.path.join(cache_dir, 'losses')
    np.save(path, losses)


def load_model_from_cache(model, cache_dir, model_file_name):
    """
    Loads model and values of the loss function from local
    files, if they exist.

    Parameters
    ----------

    model: Pytorch model

    model_file_name: str
        Name of the file in which model parameters are stored.

    other parameters:
        See q2_variables.md.

    Returns
    -------

    losses:
        See q2_variables.md.
    """

    model_path = os.path.join(cache_dir, model_file_name)
    model.load_weights(model_path)
    losses = np.load(os.path.join(cache_dir, 'losses.npy'))
    return losses


def train_model_if_not_trained(
    X, y, df, model, num_epochs, skip_epochs, learning_rate,
    cache_dir, plot_frames_dir, predictions_plot_mesh_size):
    """
    Trains the model if it has not been trained yet.

    Parameters
    ----------

    model: Pytorch model

    other parameters:
        See q2_variables.md.

    Returns
    -------

    losses:
        See q2_variables.md.
    """

    this_dir = os.path.dirname(os.path.realpath(__file__))
    full_cache_dir = os.path.join(this_dir, cache_dir)
    model_file_name = "model"
    model_path = os.path.join(full_cache_dir, model_file_name)
    model_index = f"{model_path}.index"

    if os.path.exists(model_index):
        losses = load_model_from_cache(model, cache_dir, model_file_name)
    else:
        losses = train_model(
            X=X, y=y, df=df, model=model, num_epochs=num_epochs,
            skip_epochs=skip_epochs,
            learning_rate=learning_rate,
            plot_frames_dir=plot_frames_dir,
            predictions_plot_mesh_size=predictions_plot_mesh_size
        )

        save_model_to_cache(model, losses, cache_dir, model_file_name)

    return losses


def entry_point():
    """
    Ready? Go!
    The entry point of the program.
    """

    np.random.seed(0)
    tf.random.set_seed(0)
    plot_dir = 'plots/q3_tensorflow'
    plot_frames_dir = 'plots/q3_tensorflow/movie_frames'
    n_hidden = 3
    num_epochs = 30000
    skip_epochs = 100
    predictions_plot_mesh_size = 300
    learning_rate = 1e-3

    X, y, df = read_data('data/ps5_data.csv')
    n_inputs = X.shape[1]

    # Construct the model
    model = keras.Sequential([
        layers.InputLayer(n_inputs, name="input"),
        layers.Dense(n_hidden, activation="sigmoid", name="hidden"),
        layers.Dense(1, name="output"),
    ])

    losses = train_model_if_not_trained(
        X=X, y=y, df=df, model=model, num_epochs=num_epochs,
        skip_epochs=skip_epochs,
        learning_rate=learning_rate,
        plot_frames_dir=plot_frames_dir,
        cache_dir='model_cache/q3_tensorflow',
        predictions_plot_mesh_size=predictions_plot_mesh_size
    )

    plot_losses(losses, skip_epochs, plot_dir=plot_dir)

    plot_predictions(
        X, df,
        mesh_size=predictions_plot_mesh_size,
        epoch=int(num_epochs/skip_epochs),
        image_format='pdf',
        plot_dir=plot_dir,
        run_model_func=calculate_model_output,
        run_model_args={
            "model": model,
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
