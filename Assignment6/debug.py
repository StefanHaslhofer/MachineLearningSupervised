#NOTE#######################################################################################################
#Please add all your imports in this cell only
############################################################################################################

#Nothing to do here
import numpy as np
import pandas as pd
import sys
import time
import numpy as np
import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
from mnist_loader import MNIST
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
# Set random seed to ensure reproducible runs
RSEED = 10

#Load training and test data (routine from previous assignment)
data = MNIST('./dataset/')
img_train, labels_train = data.load_training()
x_train = np.array(img_train)
y_train = np.array(labels_train)
x_test,y_test = data.load_testing()
x_test = np.array(x_test)
y_test = np.array(y_test)
print(y_train)
print(y_test)


def _filter_(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        labels_list: list[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function filters the datasets w.r.t to given labels_list.
    So, in the end x_train and x_test only contain the samples with labels that are in labels_list and y_train and y_test only contain 2 labels (i.e. 0 and 1).

    Parameters
    ----------
    x_train : np.ndarray
        Training data matrix.
    y_train : np.ndarray
        Training labels vector.
    x_test : np.ndarray
        Test data matrix.
    y_test : np.ndarray
        Test labels vector.
    labels_list : list[int, int]
        list of length 2 which consists of integer labels.

    Returns
    -------
    tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        Returns the filtered train and test data matrix and labels vector (which only conists of 0s and 1s now).
        Check the return statement to see the actual order of returned arrays.
    """
    # Your code goes here ↓↓↓
    mask = [True if y in labels_list else False for y in y_train]
    x_train_filtered = x_train[mask]
    y_train_filtered = y_train[mask]
    y_train_filtered = np.array([1 if y == 3 else 0 for y in y_train_filtered])

    mask = [True if y in labels_list else False for y in y_test]
    x_test_filtered = x_test[mask]
    y_test_filtered = y_test[mask]
    y_test_filtered = np.array([1 if y == 3 else 0 for y in y_test_filtered])
    # Your code ends here _____________________________________________________________________________________

    return x_train_filtered, y_train_filtered, x_test_filtered, y_test_filtered

x_train, y_train, x_test, y_test = _filter_(x_train, y_train, x_test, y_test, [3,8])
print(y_train)
print(y_test)


def fit_predict(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        rseed: int
) -> tuple[RandomForestClassifier, np.ndarray]:
    """Function fits a RandomForestClassifier on the training data and returns predictions as well as model.

    Parameters
    ----------
    x_train : np.ndarray
        Training data matrix.
    y_train : np.ndarray
        Training labels vector.
    x_test : np.ndarray
        Test data matrix.
    y_test : np.ndarray
        Test labels vector.
    rseed : int
        Random Seed used for initializing the Classifier.

    Returns
    -------
    tuple(RandomForestClassifier, np.ndarray)
        Where the classifier is the already trained classifier and the array is the predictions on x_test.
    """
    # Your code goes here ↓↓↓
    clf = RandomForestClassifier(random_state=rseed)
    model = clf.fit(x_train, y_train)

    prediction = model.predict(x_test)
    # Your code ends here _____________________________________________________________________________________

    return model, prediction


model, prediction = fit_predict(x_train,y_train,x_test,y_test, RSEED)


def get_n_items_wrong(
        x_test:
        np.ndarray, y_test:
        np.ndarray,
        prediction: np.ndarray
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the size of the test set, number of misclassified samples, the samples that are misclassified,
    the actual labels that are misclassified in the prediction, and the labels that are wrong in the prediction.

    Parameters
    ----------
    x_test : np.ndarray
        Test data matrix.
    y_test : np.ndarray
        Test labels vector.
    prediction : np.ndarray
        The predicted labels vector from the model.

    Returns
    -------
    tuple(int, int, np.ndarray, np.ndarray, np.ndarray)
        Where the first two ints are the size_test and num_wrong and the arrays are the samples_wrong, labels_wrong and predictions_wrong.
    """
    # Your code goes here ↓↓↓
    size_test = len(x_test)
    # create mask via np.not_equal
    mask = np.not_equal(y_test, prediction)
    num_wrong = len(prediction[mask])
    samples_wrong = x_test[mask]
    labels_wrong = y_test[mask]
    predictions_wrong = prediction[mask]
    # Your code ends here _____________________________________________________________________________________

    return size_test, num_wrong, samples_wrong, labels_wrong, predictions_wrong

#SOLUTION TO NUMBER OF WRONG PREDICTIONS
#Your code goes here ↓↓↓
size_test, num_wrong, samples_wrong, labels_wrong, predictions_wrong = get_n_items_wrong(x_test, y_test, prediction)
#Your code ends here _____________________________________________________________________________________

#Following print statement might be evaluated
print("Number of test samples: {0}\nNumber of misclassified samples: {1}".format(size_test,num_wrong))


# PLOTTING HEATMAPS
def plot_heatmaps(x_train: np.ndarray, y_train: np.ndarray, model: RandomForestClassifier) -> plt.Figure:
    """Create plots as described in the task description above.

    Parameters
    ----------
    x_train : np.ndarray
        Training data matrix.
    y_train : np.ndarray
        Training labels vector.
    model : RandomForestClassifier
        Your already trained classifier.

    Returns
    -------
    plt.Figure
        A matplotlib.pyplot figure object (i.e. the heatmaps).
    """
    # Your code goes here ↓↓↓

    # Use your freestlye plotting

    # Step 1 Split and calculate averages
    dress_av = np.reshape(np.mean(x_train[y_train == 1], axis=0), (28, 28))
    bag_av = np.reshape(np.mean(x_train[y_train == 0], axis=0), (28, 28))

    # Step 2 Subtract
    diff = dress_av - bag_av

    # Step 3 Extract feature importance
    importances = np.reshape(model.feature_importances_, (28, 28))

    # Step 4 plot everything together
    fig, axs = plt.subplots(2,2,figsize=(14, 14), gridspec_kw = {'wspace':0.15,'hspace':0.3})
    # bag
    sns.heatmap(bag_av, ax=axs[0, 0])
    axs[0, 0].set_title('Average bags')

    # dress
    sns.heatmap(dress_av, ax=axs[0, 1])
    axs[0, 1].set_title('Average dresses')

    # diff
    sns.heatmap(diff, ax=axs[1, 0])
    axs[1, 0].set_title('Overlaps')

    # importance
    sns.heatmap(importances, ax=axs[1, 1])
    axs[1, 1].set_title('Feature importances')
    plt.show()
    # Your code ends here _____________________________________________________________________________________

    return fig

fig = plot_heatmaps(x_train, y_train, model)