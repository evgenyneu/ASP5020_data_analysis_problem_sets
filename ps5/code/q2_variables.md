
# Common variables used in q2 code.


## gradients

numpy.ndarray 1D array.
Contains the list of gradients for all weights.

Example:

For network with two inputs and three nodes in the hidden layer:

```
[g1, g2, g3, g4, g5, g6, .... g13],
```

where g1, g2, g3 are gradients for the weights for connections coming from the bias in the input layer, g4, g5, g6 are for first input node etc.


## hidden_layer_outputs

N by H numpy.ndarray 2D array.
Outputs coming from the nodes of the hidden layer.
where
  * N is the number of observations,
  * H is the number nodes in the hidden layer (excluding the bias)

Example:

For three nodes with outputs h1, h2, h3 in the hidden layer:
```
[
    [h1, h2, h3],
    [h1, h2, h3],
    [h1, h2, h3],
    ...
    [h1, h2, h3]
].
```

## hidden_layer_weights

M + 1 by H numpy.ndarray 2D array.
Weights for connections coming from the input into the hidden layer,
where
  * M is the number of input variables,
  * H is the number nodes in the hidden layer (excluding the bias).

Example:

For two input variables and three hidden nodes:
```
[
    [w1,  w2,  w3],
    [w4,  w5,  w6],
    [w7,  w8,  w9]
].
```
Here weights w1, w2 and w3 are for connections coming from input layer bias, w4, 25 and w6 for first input x1 and w7, w8 and w9 are for the second input.


## inputs_with_bias

N by M + 1 numpy.ndarray 2D array.
Input layer values that include the bias 1 and input variables for all observations, where
  * N is the number of observations,
  * M is the number of input variables.

Example:

For two variables x1 and x2, this array looks like:
```
[
    [1  x1 x2],
    [1  x1 x2],
    [1  x1 x2]
    ...
    [1  x1 x2]
].
```


## losses

list of floats
Values of the loss functions as successive epochs.



## n_inputs

int
Number of input variables.


## n_hidden

int
Number of nodes in the hidden layer (excluding the bias node).


## n_observations

int
Number of observations.


##  output_layer_weights

H + 1 by 1 numpy.ndarray 2D array,
Weights for connections coming from the hidden layer into the output node,
where
  * H is the number nodes in the hidden layer (excluding the bias).

Example:

For a hidden layer with three nodes:
```
[
    [w10],
    [w11],
    [w12],
    [w13]
],
```
where w10 is for connection coming from hidden layer bias, and w11, w12, w13
are for three nodes in the hidden layer.


## predictions_plot_mesh_size:

int
Number of predictions to display along each of the two axes
of prediction plot.



## x

N by M numpy.ndarray 2D array
Input data values, where
  * N is the number of observations,
  * M is the number of input variables.

Example:

For two inputs x1 and x2:
```
[
    [x1, x2]
    [x1, x2]
    [x1, x2]
    ...
    [x1, x2]
]
```

## X

Same as x but original, not normalized inputs.


## y

N by 1 numpy.ndarray 2D array.
Contains output data values for N observations.


## y_pred

N by 1 numpy.ndarray 2D array.
Contains model predictions for N observations.
