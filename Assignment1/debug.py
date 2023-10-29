import sklearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.collections import PathCollection
from sklearn.model_selection import KFold
from sklearn import neighbors


def add_features(X):
    """
    Function that adds random features to dataset
    @param X, np array, dataset

    @output X_new, np array, dataset enhanced with 4 random features
    """
    np.random.seed(1234)
    # your code goes here ↓↓↓
    # generate matrix of random numbers with 4 columns
    C = np.random.rand(X.shape[0], 4)
    # append C to X along column axis
    X_new = np.append(X, C, axis=1)

    for i in range(2,6):
        print(i)
    return X_new


m = 179


def plot_error_vs_k_extra_dims(X_new, y, m):
    """function that implements the plot from Task5.

    This function should create 2! plots.
    The first plot to visualize error vs k with 10 folds and <X> extra dimensions.
    The second plot to visualize the error versus dimension with k=11

    Returns
    -------
    tuple[Figure, Figure]
        A tuple with 2! matplotlib figures
    """
    # your code goes here ↓↓↓
    fig1, axs = plt.subplots(2, 2, figsize=(14, 14), gridspec_kw={'wspace': 0.15, 'hspace': 0.3})
    # implement first plot
    # 1 extra dim
    error_holder = []

    fig2 = plt.figure(figsize=(12, 6))
    # implement second plot

    return fig1, fig2

Z = np.genfromtxt('DataSet1.csv', delimiter=';')
X, y = Z[:,:-1], Z[:,-1]
X_new = add_features(X)