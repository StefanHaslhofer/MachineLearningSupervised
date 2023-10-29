import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random

# read data, split into X (features) and y (labels)
Z = np.genfromtxt('normal.csv', delimiter=',')
X, y = Z[:,:-1], Z[:,-1]


## your code goes here ↓↓↓
def scatter_plot(X, y):
    """creates a scatter-plot for the dataset X with labels y

    Parameters
    ----------
    X : np.ndarray
        data
    y : np.ndarray
        labels

    Returns
    -------
    Figure
        a matplotlib figure object
    """
    fig1 = plt.figure(figsize=(8, 5))
    # your code goes here ↓↓↓
    df = pd.DataFrame(dict(
        x1=X[:, 0],
        x2=X[:, 1],
        label=y
    ))

    for name, group in df.groupby('label'):
        plt.scatter(group.x1, group.x2, label=name)

    plt.title('DataSet1', fontweight='bold')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    return fig1

# Nothing to do here, just run the cell.
fig = scatter_plot(X, y)
fig.show()
assert isinstance(fig, Figure)