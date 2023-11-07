import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random


# Nothing to do here, just run the cell.

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# do not change the seed
np.random.seed(12)

def pol_reg_pred(X_train,y_train,X_test,m):
    """
    Function that trains a polynomial regression model with degree on a given training set
    and returns the prediction for a given test set (uniformly sampled values without labels).
    @param X_train, np.ndarray, training samples
    @param y_train, np.ndarray, training labels
    @param X_test, np.ndarray, test samples
    @param m, int, degree of polynomial
    """
    np.random.seed(12)
    poly_reg = PolynomialFeatures(m)
    X_poly_train = poly_reg.fit_transform(X_train.reshape(-1, 1))
    X_poly_test= poly_reg.fit_transform(X_test.reshape(-1, 1))
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly_train, y_train)
    y_pred = lin_reg.predict(X_poly_test)
    return y_pred


def func_f(x):
    """
    Implementation of the polynomial from 4.1
    @param x, value you pass to the function
    """
    # your code goes here ↓↓↓

    return 0.5 * pow(x, 4) + 2 * pow(x, 3) - 8 * pow(x, 2)


def create_train_X(k, l, xmin, xmax):
    """
    Function that creates k training sets with l samples
    @param k, number of training sets to create
    @param l, number of samples per training set
    @param xmin, lower bound of sample interval
    @param xmax, upper bound of sample interval
    """
    # use the numpy.random module and dont change the seed!
    np.random.seed(12)

    # your code goes here ↓↓↓
    train_x = np.random.uniform(low=xmin, high=xmax, size=(k, l))
    return train_x


def create_train_y(k, l, X_train, mu, std, func):
    """
    Function that creates labels from training data with func_f and gaussian noise
    @param k, number of label sets to create
    @param l, number of labels per set
    @param X_train, training set
    @param mu, mean of gaussian
    @param std, std of gaussian
    @param func, callable, polynomial function
    returns np.ndarray
    """
    # use the numpy.random module and dont change the seed!
    np.random.seed(12)

    # your code goes here ↓↓↓
    noise = np.random.normal(mu, std, size=(k, l))
    train_y = np.zeros((k, l))

    for k_i in range(k):
        for l_i in range(l):
            train_y[k_i][l_i] = func(X_train[k_i][l_i]) + noise[k_i][l_i]

    return train_y


def bias_var(X_train, y_train, M, k, func):
    """
    Function that computes model bias and variance
    @param X_train, np.ndarray, training data
    @param y_train, np.ndarray, training labels
    @param M, int, upper bound on m (degree of the polynomial)
    @param k, int, number of sample sets
    @param func, callable, polynomial function
    returns sqbias,variance
    """
    x0 = np.array([1.7])
    sqbias = []
    variance = []

    # your code goes here ↓↓↓
    for m_i in range(M):
        y_pred = []
        for k_i in range(k):
            y_pred.append(pol_reg_pred(X_train[k_i], y_train[k_i], x0, m_i + 1))
        sqbias.append((func(x0[0]) - np.mean(y_pred))** 2)
        variance.append(np.mean((y_pred - np.mean(y_pred))** 2))

    return (sqbias, variance)

## If you get deprecation warnings from numpy in the following cell, uncomment these two lines:
#import warnings
#warnings.filterwarnings('ignore')
## Else: Nothing to do here, just run the cell.
k = 300
l = 25
M = 11
xmin=-1
xmax=3
mu=0
sigmasq=4
std=np.sqrt(sigmasq)

X_train = create_train_X(k,l,xmin,xmax)
y_train = create_train_y(k,l,X_train,mu,std,func_f)
sqbias, variance = bias_var(X_train,y_train,M,k,func_f)

print("Shapes of X and y: \n",X_train.shape,y_train.shape)
print("\nSquared Bias over m: \n", sqbias)
print("\nVariance over m: \n", variance)

## test data, from -1 to 3 in steps of 0.01
# Nothing to do here, just run the cell.
np.random.seed(12)
x_ = np.arange(xmin, xmax, 0.01)   ## test data, from -1 to 3 in steps of 0.01


def plot3(X_train, y_train, x, pol_reg_):
    """creates a plot for the training data and the corresponding regression models with different m

    Parameters
    ----------
    X_train : np.ndarray
        training data
    y_train : np.ndarray
        labels
    x       : np.ndarray
        test data
    pol_reg_ : function
        polynomial regression function
    Returns
    -------
    Figure
        a matplotlib figure object
    """
    fig3 = plt.figure(figsize=(8, 5))
    # your code goes here ↓↓↓
    plt.title('Polynomial Regression Prediction', fontweight='bold')
    plt.scatter(X_train[0], y_train[0])
    plt.plot(x, pol_reg_pred(X_train[0], y_train[0], x, 1), c='r', label='m=1')
    plt.plot(x, pol_reg_pred(X_train[0], y_train[0], x, 3), c='y', label='m=3')
    plt.plot(x, pol_reg_pred(X_train[0], y_train[0], x, 11), c='m', label='m=11')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    return fig3


def plot4(M, sigmasq, biassq, var):
    """creates a plot for  unavoidable error, bias, variance and EPE vs. m in [1,11]

    Parameters
    ----------
    M    : int
        m is in [1,M]
    sigmasq: float
        unavoidable error
    biassq : np.ndarray
        squared bias
    var  : np.ndarray
        model variance
    Returns
    -------
    Figure
        a matplotlib figure object
    """
    fig4 = plt.figure(figsize=(8, 5))
    # your code goes here ↓↓↓
    plt.title('Error to degree of the polynomial', fontweight='bold')
    plt.plot(range(M), np.full(M, sigmasq), c='r', label='unavoidable error')
    plt.plot(range(M), biassq, c='b', label='squared bias')
    plt.plot(range(M), var, color='orange', label='variance')
    plt.plot(range(M), np.sum([biassq, var, np.full(M, sigmasq)], axis=0), c='g', label='total EPE', ls='--')
    plt.xlabel("m")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
    return fig4

fig4 = plot4(M, sigmasq, sqbias, variance)
assert isinstance(fig4, Figure)