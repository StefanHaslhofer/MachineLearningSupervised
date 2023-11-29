import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random


Z = np.genfromtxt('normal.csv', delimiter=',')
X, y = Z[:,:-1], Z[:,-1]

def est_mean_cov(X_, y_):
    """
    Function that estimates the means and covariance matrices from the given data as well
    as the probability to encounter a positive/negative example respectively
    @param X_, np ndarray, data matrix
    @param y_, np ndarray, data vector
    Returns
    covX, covXpos, covXneg: covariance matrices for entire dataset, positive samples, negative samples
    meanX, meanXpos, meanXneg: means for entire dataset, positive samples, negative samples
    p_ypos, p_yneg: probabilites p(y=+1), p(y=-1)
    """
    # your code goes here ↓↓↓
    groups = pd.DataFrame(dict(
        x1=X_[:, 0],
        x2=X_[:, 1],
        label=y_
    )).groupby('label')

    # means
    mean = groups.mean()
    meanX = [
        sum(mean.x1.values) / len(mean.x1),
        sum(mean.x2.values) / len(mean.x2)
    ]
    meanXpos = [mean.x1[1], mean.x2[1]]
    meanXneg = [mean.x1[-1], mean.x2[-1]]

    # covariance
    cov = groups.cov()

    covX = pd.DataFrame(dict(
        x1=X_[:, 0],
        x2=X_[:, 1]
    )).cov()
    covXpos = [
        [cov.x1[1].x1, cov.x1[1].x2],
        [cov.x2[1].x1, cov.x2[1].x2]
    ]
    covXneg = [
        [cov.x1[-1].x1, cov.x1[-1].x2],
        [cov.x2[-1].x1, cov.x2[-1].x2]
    ]

    # probabilities
    p_ypos = len(groups.get_group(1)) / len(y_)
    p_yneg = 1 - p_ypos

    return (meanX, covX, meanXpos, covXpos, meanXneg, covXneg, p_ypos, p_yneg)

# Nothing to do here, just run the cell.
meanX, covX, meanXpos, covXpos, meanXneg, covXneg, p_ypos, p_yneg = est_mean_cov(X,y)


def calc_par_A(meanXpos, covXpos, meanXneg, covXneg, p_ypos, p_yneg):
    """
    This function should contain the calculations for the respective parameter and return the result.
    @param meanXpos, np ndarray, mean of positive examples
    @param covXpos, np ndarray, covariance matrix of positive examples
    @param meanXneg, np ndarray, mean of negative examples
    @param covXneg, np ndarray, covariance matrix of negative examples
    @param p_ypos, float, probability of encountering a positive example
    @param p_yneg, float, probability of encountering a negative example
    returns np.ndarray
    """
    # your code goes here ↓↓↓
    par_A = np.linalg.inv(covXpos) - np.linalg.inv(covXneg)

    return par_A

def calc_par_b(meanXpos,covXpos,meanXneg,covXneg,p_ypos,p_yneg):
    """
    This function should contain the calculations for the respective parameter and return the result.
    @param meanXpos, np ndarray, mean of positive examples
    @param covXpos, np ndarray, covariance matrix of positive examples
    @param meanXneg, np ndarray, mean of negative examples
    @param covXneg, np ndarray, covariance matrix of negative examples
    @param p_ypos, float, probability of encountering a positive example
    @param p_yneg, float, probability of encountering a negative example
    returns np.ndarray
    """
    #your code goes here ↓↓↓
    par_b = np.matmul(np.linalg.inv(covXpos), meanXpos) - np.matmul(np.linalg.inv(covXneg), meanXneg)

    return par_b


def calc_par_c(meanXpos, covXpos, meanXneg, covXneg, p_ypos, p_yneg):
    """
    This function should contain the calculations for the respective parameter and return the result.
    @param meanXpos, np ndarray, mean of positive examples
    @param covXpos, np ndarray, covariance matrix of positive examples
    @param meanXneg, np ndarray, mean of negative examples
    @param covXneg, np ndarray, covariance matrix of negative examples
    @param p_ypos, float, probability of encountering a positive example
    @param p_yneg, float, probability of encountering a negative example
    returns np.float64
    """
    # your code goes here ↓↓↓
    c1_neg = -(1 / 2) * np.matmul(np.transpose(meanXpos), np.matmul(np.linalg.inv(covXpos), meanXpos))
    c1_pos = +(1 / 2) * np.matmul(np.transpose(meanXneg), np.matmul(np.linalg.inv(covXneg), meanXneg))

    det_neg = -(1 / 2) * np.log(np.linalg.det(covXpos))
    det_pos = (1 / 2) * np.log(np.linalg.det(covXneg))

    par_c = c1_neg + c1_pos + det_neg + det_pos + np.log(p_ypos) - np.log(p_yneg)

    return par_c


def calc_func_g(par_A, par_b, par_c, gridpoints):
    """
    Combine the previously calculated parameters to the optimal classification function g.
    Return in shape [500,500]. The 500x500 grid will plot nicely later.
    Avoid hardcoding, i.e. use int(np.sqrt(gridpoints.shape[0]) instead of the number 500
    @param gridpoints, np.array, the points the function g should be applied to
    returns np.ndarray of shape (500,500)
    """
    # your code goes here ↓↓↓
    g = np.empty((int(np.sqrt(gridpoints.shape[0])), int(np.sqrt(gridpoints.shape[0]))))
    for x1 in range(g.shape[0]):
        for x2 in range(g.shape[1]):
            x_vec = gridpoints[x1 * 500 + x2]
            g[x1][x2] = (-(1 / 2) * np.matmul(np.transpose(x_vec), np.matmul(par_A, x_vec)) + np.matmul(
                np.transpose(par_b), x_vec) + par_c)

    return np.sign(g)


# Nothing to do here, just run the cell.

X1, X2 = np.mgrid[-11:11:500j, -11:11:500j]
gridpoints = np.c_[X1.ravel(), X2.ravel()]

par_A = calc_par_A(meanXpos, covXpos, meanXneg, covXneg, p_ypos, p_yneg)
par_b = calc_par_b(meanXpos, covXpos, meanXneg, covXneg, p_ypos, p_yneg)
par_c = calc_par_c(meanXpos, covXpos, meanXneg, covXneg, p_ypos, p_yneg)
func_g = calc_func_g(par_A, par_b, par_c, gridpoints)
print("gridponts.shape =", gridpoints.shape, "\n")
print("func_g.shape =", func_g.shape, "\n")
print("A = ", par_A)
print("A.shape = ", par_A.shape, "\n")
print("b = ", par_b)
print("b.shape = ", par_b.shape, "\n")
print("c = ", par_c)
print("c.shape = ", par_c.shape)


# Visualize the data and the classifier with a scatter plot
def scatter_plot2(X, X1, X2, y, g):
    """Creates a scatter-plot for the dataset X with labels y and the classification function g
    Parameters
    ----------
    X : np.ndarray
        data
    X1: np.ndarray
        grid x values
    X2: np.ndarray
        grid y values
    y : np.ndarray
        labels
    g : np.ndarray
        the matrix from your Gaussian classifier
    Returns
    -------
    Figure
        a matplotlib figure object
    """
    fig2 = plt.figure(figsize=(8, 5))
    # your code goes here ↓↓↓
    fig2 = plt.figure(figsize=(8, 5))
    # your code goes here ↓↓↓
    df = pd.DataFrame(dict(
        x1=X[:, 0],
        x2=X[:, 1],
        label=y
    ))




    plt.scatter(X1[1 == g[:]], X2[1 == g[:]], s=1, alpha=0.05, zorder=-1)
    plt.scatter(X1[-1 == g[:]], X2[-1 == g[:]], s=1, alpha=0.05, zorder=-1)
    plt.show()

    return fig2

# Nothing to do here, just run the cell.
g = calc_func_g(par_A,par_b,par_c,gridpoints)
fig = scatter_plot2(X, X1, X2, y, g)
assert isinstance(fig, Figure)