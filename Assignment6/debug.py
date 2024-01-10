import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple

# set the seed for reproducability to 44
RSEED = 44
np.random.seed(RSEED)


def logistic_gradient(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Function that computes the logistic gradient via the cross-entropy loss.

    Parameters
    ----------
    w : np.ndarray
        weights
    x : np.ndarray
        data matrix
    y : np.ndarray
        data labels

    Returns
    -------
    np.ndarray
        gradient vector
    """
    gradient = np.ndarray(shape=np.shape(w))

    for j in range(len(w)):
        for i in range(len(y)):
            sigma_i = 1 / (1 + np.exp(-(np.dot(w, x[i]))))
            gradient[j] += (sigma_i - y[i]) * x[i][j]

    return gradient


def generate_random(nr_samples: int, nr_features: int, seed=RSEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Function that generates a random matrix X and the random vectors y and weights according to input specifications.
    Hint: to generate the distributions use numpy's np.random.<...> functions.

    Parameters
    ----------
    nr_samples : int,
        number of samples to generate
    nr_features : int,
        number of features of each sample
    seed: int, optional
        seed for np.random, dont change it!
    Returns
    -------
    tuple(np.ndarray,np.ndarray,np.ndarray)
        returns randomly generated data matrices X,y and w.
    """
    # don't change the seed!
    np.random.seed(seed)

    # Your code goes here ↓↓↓
    X_random = np.random.normal(size=(nr_samples, nr_features))
    y_random = np.random.randint(size=(nr_samples), low=0, high=2)
    w_random = np.random.normal(size=(nr_features))

    return (X_random, y_random, w_random)


def numerical_gradient(w: np.ndarray, x: np.ndarray, y: np.ndarray, eps_list: list, cost_function: callable) -> list:
    """Function that computes the numerical gradient for each epsilon in eps_list.
    Hint: Use the previously implemented cost function, it is given as input to this function,
    make sure to call it with the correct name (cost_function(...), not cost(...))!

    Parameters
    ----------
    w : np.ndarray
        weights
    x : np.ndarray
        data matrix
    y : np.ndarray
        data labels
    eps : list
        list of floats; the epsilon values for difference quotient calculation
    cost_function: callable
        function that calculates the cross-entropy loss, implemented by you

    Returns
    -------
    list
        list of same length as eps_list containing the np.ndarrays from the computed gradients for each epsilon in eps_list
    """
    dw_list = []

    # Your code goes here ↓↓↓

    return dw_list


def cost(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Function that computes the cross-entropy loss

    Parameters
    ----------
    w : np.ndarray
        weights
    x : np.ndarray
        data matrix
    y : np.ndarray
        data labels

    Returns
    -------
    float
        cross-entropy loss
    """
    loss = None
    # Your code goes here ↓↓↓

    return loss


def comparison(grad_a: np.ndarray, grad_n: np.ndarray) -> bool:
    """Function that compares two arrays, in our case the numerical and analytical gradients.
    Hint: use absolute tolerance, with an error tolerance of 1e-7.

    Parameters
    ----------
    grad_a : np.ndarray
        the analytical gradient
    grad_n : np.ndarray
        the numberical gradient

    Returns
    -------
    bool
        True, if arrays are close, according to our specifications, False if not.
    """
    close = np.all(np.abs(grad_a - grad_n) <= 1e-7)

    return close


# Nothing to do here, if you did everything correctly, you can just run this code and should see the correct results

n = 5  # number of samples
d = 10  # number of features
eps_list = [1e-1, 1e-4, 1e-11]  # epsilon values to test
X_random, y_random, w_random = generate_random(n, 10, RSEED)
analytical_gradient = logistic_gradient(w_random, X_random, y_random)
print("Logistic gradient:\n", analytical_gradient, "\n")
num_gradients = numerical_gradient(w_random, X_random, y_random, eps_list, cost)
comparison_results = []
for grad in num_gradients:
    comparison_results.append(comparison(analytical_gradient, grad))
# Check outputs
assert len(comparison_results) == len(
    eps_list), "List with comparison results should be the same length as there are epsilon values."
assert isinstance(comparison_results[0], bool), f"Comparison results should be list of booleans."
print("Your randomly generated data:\n")
print("X =", X_random, "\n")
print("y =", y_random, "\n")
print("w = ", w_random, "\n")
print("Logistic gradient:\n", analytical_gradient, "\n")
for i, res in enumerate(comparison_results):
    eps_ = eps_list[i]
    print(f"Numerical gradient {i} with epsilon = {eps_}:\n", num_gradients[i], "\n")
    print("    Vectors within absolute tolerance of 10^-7: ", res, "\n")
