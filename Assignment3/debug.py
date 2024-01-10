#Nothing to do here
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from typing import Sequence, List, Tuple

np.random.seed(1234)  

import warnings
warnings.filterwarnings("ignore")

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

from sklearn import svm


# Nothing to do here
def load_data(id_data: int = 1):
    """Function allows to load data from csv
    @returns: tuple (X,y)"""
    if id_data == 1:
        Z = np.genfromtxt('radial_data.csv', delimiter=',')
        return Z[:, :-1], Z[:, -1]


def get_meshgrid(X, resolution):
    """Function creates space/grid. Mostly used for plotting"""
    s = np.max(np.abs(X)) * 1.05
    ls = np.linspace(-s, s, resolution)
    X1, X2 = np.meshgrid(ls, ls, sparse=False)
    return np.c_[X1.ravel(), X2.ravel()]


def plot_data(X, y,
              model=None,
              plot_boarders=True,
              plot_classification=True,
              plot_support_vectors=True,
              plot_size=7,
              resolution=500,
              title='data visualization',
              color=['blue', 'orange']):
    """Plotting your data
    @param model already trained SVM model, is None if you want to plot data only
    all other parameters must be intuitively clear for you"""

    if model is not None:  # if you want to plot model
        if plot_classification and plot_boarders:
            col = 2  # if you want to plot model and boarders
        else:
            col = 1  # if you want to plot model only

        fig, axs = plt.subplots(1, col, figsize=(plot_size * col, plot_size))

        grid = get_meshgrid(X, resolution)
        V = model.support_vectors_
        mask_sv = model.support_  # np.where(np.isin(X[:,0],V[:,0]))[0]

        kernel = model.kernel
        if kernel == 'poly':
            title = f"kernel: {kernel} - degree: {model.degree} - cost:{model.C}"
        elif kernel == 'rbf':
            title = f"kernel: {kernel}"
            if model.gamma != "auto_deprecated":
                title += f" - gamma: {model.gamma}"
            title += f" - cost: {model.C}"

        for i in range(col):
            if col > 1:
                ax = axs[i]
            else:
                ax = axs
            ax.set_aspect('equal')
            if i == 0 and plot_boarders:
                ax.set_title('Margins - ' + title, fontsize=plot_size * 2)
                boarders = model.decision_function(grid)
                mask_pos = boarders >= 1
                mask_neg = boarders <= -1
                ax.scatter(grid[mask_pos, 0], grid[mask_pos, 1], c=color[0], alpha=0.01, s=10)
                ax.scatter(grid[mask_neg, 0], grid[mask_neg, 1], c=color[1], alpha=0.01, s=10)
                ax.scatter(X[mask_sv, 0], X[mask_sv, 1], c='g', label=str(np.sum(model.n_support_)) + ' SV', s=40,
                           marker='o')
            if plot_classification and (i == 1 or not plot_boarders):
                ax.set_title('Classification - ' + title, fontsize=plot_size * 2)
                classification = model.predict(grid)
                mask_pos = classification > 0
                mask_neg = classification < 0
                ax.scatter(grid[mask_pos, 0], grid[mask_pos, 1], c=color[0], alpha=0.01, s=10)
                ax.scatter(grid[mask_neg, 0], grid[mask_neg, 1], c=color[1], alpha=0.01, s=10)
                classification = model.predict(X)
                mask_wrong = classification != y
                ax.scatter(X[mask_wrong, 0], X[mask_wrong, 1], c='magenta', label=str(np.sum(mask_wrong)) + ' faults',
                           s=40, marker='o')
            m = y > 0
            ax.scatter(X[m, 0], X[m, 1], c=color[0], label='class +1', s=10)
            m = np.logical_not(m)
            ax.scatter(X[m, 0], X[m, 1], c=color[1], label='class -1', s=10)
            ax.legend(loc='lower left', fontsize=plot_size * 1.5)

    else:
        fig, axs = plt.subplots(1, 1, figsize=(plot_size, plot_size))
        axs.set_aspect('equal')
        m = y > 0
        axs.scatter(X[m, 0], X[m, 1], c=color[0], label='class +1', s=10)
        m = np.logical_not(m)
        axs.scatter(X[m, 0], X[m, 1], c=color[1], label='class -1', s=10)
        axs.legend(loc='lower left', fontsize=plot_size * 1.5)
        plt.title(title, fontsize=plot_size * 2)
    plt.show()

#load data
#leave as it is
X,y = load_data(1)

print(X)

def pol2cart(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return(x, y)

def create_new_data() -> Tuple[np.ndarray, np.ndarray]:
    """Function that creates the new data

    For random numbers use np.random module, not np.random.default_rng

    Hint
    ----
    use pol2cart

    Returns
    -------
    tuple
        tuple of 100 2d datapoints and labels (X_new, y_new)
    """
    # Your solution here:
    n = 100
    rad = 0.25
    # phi ranges from 0 to 2*pi
    # (full rotation is 2*pi)
    phi = np.random.uniform(0, 2 * np.pi, n)
    # distance from center is in [0, rad]
    r = np.random.uniform(0, rad, n)

    x1, x2 = pol2cart(r, phi)
    y_new = np.full(100, -1)

    X_new = np.stack((x1, x2), axis=1)
    return X_new, y_new

X_new, y_new = create_new_data()

X_new = np.concatenate((X, X_new), axis=0)
y_new = np.concatenate((y, y_new), axis=0)
plot_data(X_new, y_new)

def iter_Gamma(gamma_range: Sequence[float], C: float, X_new: np.ndarray, y_new: np.ndarray) -> List[dict]:
    """
    Function iter_Gamma fits SVM using defined gamma range and plots every variation.
    @ gamma_range: list of gammas
    @ returns: list of dictionaries, length of list = length of gamma_range
    """
    list_of_model_parameters = []
    # your code ↓↓↓
    # code ends here
    for gamma in gamma_range:
        model = svm.SVC(gamma=gamma, C=C)
        model.fit(X_new, y_new)
        list_of_model_parameters.append(model.get_params())
        plot_data(X_new, y_new, model)

    return list_of_model_parameters

# update X_new and y_new by adding point as described in the task↓↓↓
def add_points(X_new: np.ndarray, y_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # your code here
    X_new_ = np.append(X_new, [[1.8, 1.3]], axis=0)
    y_new_ = np.append(y_new, 1)
    return X_new_, y_new_

# nothing to do, just run the cell
X_new_plus1, y_new_plus1 = add_points(X_new, y_new)

# This cell is just to discover the behaviour of SVM given the extra point, not graded.

cost_values = range(-1,4)
gamma_values = [0.1,0.5,0.9]

for cost in cost_values:
    iter_Gamma(gamma_values,10**cost,X_new_plus1,y_new_plus1)