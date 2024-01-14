# Python Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
import os

# Notes on shapes of matrices:
# the pre-activations and activations should be row-vectors
# meaning shape = (1, ...). Think of it as having a mini-batch size of 1
# Check out the pytorch docs on the linear layer for the forward pass and the shape
# of the weight-layers ;)

input = np.array([1, 1.5]).reshape(1, -1)

weights_layer1 = np.array([[-0.1, 0.5], [-1, 0], [0.1, -2]])  # fill in the correct values
weights_layer2 = np.array([[1.2, -1, 0.5]])  # fill in the correct values

assert weights_layer1.shape == (3, 2)
assert weights_layer2.shape == (1, 3)


# Note: Every function can be easily written as a small one-liner
# If you have bigger functions, maybe think of a better solution :)

def relu(x):
    """calculate relu activation of x"""
    return np.maximum(0, x)


def derivative_relu(x):
    """calculate derivative of relu for input x"""
    return 0 if x < 0 else 1 if x > 0 else None


def calc_activation(preactivation):
    """calculate activation given pre-activation
    """
    return relu(preactivation)


def calc_preactivation(x, w):
    """calculate preactivation of a linear layer

    x: layer inputs
    w: layer weights
    """
    return np.sum(x * w)


def calc_delta_6(y_hat, y):
    """delta error of last layer"""
    return y_hat - y


def calc_delta(s, delta_6, w):
    """delta error of non-last layer

    s: correct preactivation
    delta_6: delta error of last layer
    w: correct nn weight

    to get correct inputs check the formulas and their indices ;)
    """
    return derivative_relu(s) * delta_6 * w


def calc_derivative_L(delta_error, activation):
    """calc derivative of loss-fct w.r.t a certain weight.
    check formulas to now what delta_error and activation you have to
    provide :)
    """
    return delta_error * activation


x1 = 1
x2 = 1.5
w31 = -0.1
w41 = -1
w51 = 0.1
w32 = 0.5
w42 = 0
w52 = -2
w63 = 1.2
w64 = -1
w65 = 0.5

# TODO test
