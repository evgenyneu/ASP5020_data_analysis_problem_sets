from scipy.optimize import check_grad
import numpy as np

from q2_neural_network_numpy import sigmoid, make_input, reshape_weights, \
     calculate_model_output, loss_function, calculate_gradients

from pytest import approx


def test_sigmoid():
    assert sigmoid(2) == approx(0.880797, rel=1e-4)


def function(p, x, y, n_inputs, n_hidden):
    hidden_layer_inputs = make_input(x)
    n_observations = x.shape[0]
    n_hidden_weights = (n_inputs + 1) * n_hidden
    hidden_weights = p[0: n_hidden_weights]
    output_weights = p[n_hidden_weights:]

    hidden_layer_weights, output_layer_weights = reshape_weights(
        hidden_weights, output_weights, n_inputs, n_hidden)

    y_pred, _ = calculate_model_output(
        n_observations, hidden_layer_inputs,
        hidden_layer_weights, output_layer_weights)

    return loss_function(y, y_pred)


def gradient(p, x, y, n_inputs, n_hidden):
    hidden_layer_inputs = make_input(x)
    n_observations = x.shape[0]
    n_hidden_weights = (n_inputs + 1) * n_hidden
    hidden_weights = p[0: n_hidden_weights]
    output_weights = p[n_hidden_weights:]

    hidden_layer_weights, output_layer_weights = reshape_weights(
        hidden_weights, output_weights, n_inputs, n_hidden)

    y_pred, hidden_layer_outputs = calculate_model_output(
            n_observations, hidden_layer_inputs,
            hidden_layer_weights, output_layer_weights)

    gradients = np.empty(len(p))

    calculate_gradients(x, y, y_pred, hidden_layer_outputs,
                        output_layer_weights, gradients)

    return gradients


def test_test_grads():
    n_inputs = 2
    n_hidden = 3

    x = np.array(
        [
            [1, 2],
            [4, 5],
            [-2, 9]
        ])

    y = [
            [1],
            [2],
            [12]
        ]

    x0 = np.random.normal(size=13)

    grad_diff = check_grad(
        function, gradient, x0, x, y, n_inputs, n_hidden
    )

    assert grad_diff < 1e-4
