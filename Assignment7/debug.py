# Python Implementation
import numpy as np

# Notes on shapes of matrices:
# the pre-activations and activations should be row-vectors
# meaning shape = (1, ...). Think of it as having a mini-batch size of 1
# Check out the pytorch docs on the linear layer for the forward pass and the shape
# of the weight-layers ;)

input = np.array([1, 1.5]).reshape(1, -1)

weights_layer1 = ...  # fill in the correct values
weights_layer2 = ...  # fill in the correct values

assert weights_layer1.shape == (3, 2)
assert weights_layer2.shape == (1, 3)


# Note: Every function can be easily written as a small one-liner
# If you have bigger functions, maybe think of a better solution :)

def relu(x):
    """calculate relu activation of x"""
    return ...


def derivative_relu(x):
    """calculate derivative of relu for input x"""
    return ...


def calc_activation(preactivation):
    """calculate activation given pre-activation
    """
    return ...


def calc_preactivation(x, w):
    """calculate preactivation of a linear layer

    x: layer inputs
    w: layer weights
    """
    return ...


def calc_delta_6(y_hat, y):
    """delta error of last layer"""
    return ...


def calc_delta(s, delta_6, w):
    """delta error of non-last layer

    s: correct preactivation
    delta_6: delta error of last layer
    w: correct nn weight

    to get correct inputs check the formulas and their indices ;)
    """
    return ...


def calc_derivative_L(delta_error, activation):
    """calc derivative of loss-fct w.r.t a certain weight.
    check formulas to now what delta_error and activation you have to
    provide :)
    """
    return ...