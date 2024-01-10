# nothing to do here
import numpy as np
import pandas as pd
import sys
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mnist_loader import MNIST

import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
# Set random seed to ensure reproducible runs
RSEED = 50

# a simple data set for demonstration
# nothing to do here
X = np.array([[2, 2],
              [2, 1],
              [2, 3],
              [1, 2],
              [1, 1],
              [3, 3],
              [3, 2]])

y = np.array([0, 1, 1, 1, 0, 1, 0])


def train_dec_tree(x_train: np.ndarray, y_train: np.ndarray, param_dict_grid: dict, seed: int, n_iter: int, cv: int):
    """Trains a decision tree using cross-validation and returns certain attributes of the received model including the best
    parameter combination. Again use (only!) the implementations from sklearn already imported for this assignment and don't forget the seed.

    Parameters
    ----------
    x_train : np.ndarray
        data matrix
    y_train : np.ndarray
        data vector - labels
    param_dict_grid : dict
        dictionary of parameters for grid search (RandomizedSearchCV)
    seed : int
        seed for reproducability, feed to both RandomizedSearchCV and the DecisionTreeClassifier!
    n_iter : int
        number of iterations (RandomizedSearchCV)
    cv : int
        number of folds in CV (RandomizedSearchCV)

    Returns
    -------
    tuple(dict, sklearn.model_selection._search.RandomizedSearchCV)
        Best model parameters as dict and the best sklearn model fit on the training data.
        Don't forget to fit the final best model on the training data, before returning it.
    """

    classifier = None
    model = None
    model_best_params = None

    # your code ↓↓↓
    classifier = DecisionTreeClassifier(random_state=seed)
    model = RandomizedSearchCV(random_state=seed)

model = svm.SVC()
    return model_best_params, model

# print number of tree nodes and the maximum depth of tree
nr_nodes, max_depth, acc = dec_tree(RSEED,X,y)
print(f'Decision tree has {nr_nodes} nodes with maximum depth {max_depth}.')
print(f'Model accuracy: {acc}')