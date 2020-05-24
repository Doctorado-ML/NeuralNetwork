'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
Util functions to use with the classifier
'''

import numpy as np
import matplotlib.pyplot as plt


def one_hot(label, num):
    yht = np.zeros((label.size, num))
    yht[np.arange(label.size), label.T] = 1
    return yht


def plot_decision_boundary(model, X, y, binary, title):
    y = y.T[0]
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    case = np.array(np.c_[xx.ravel(), yy.ravel()])
    if type(model).__name__ == 'N_Network':
        if binary:
            Z = model.predict(case)
        else:
            Z = model.predict_proba(case)
    else:
        Z = model.predict(case)
        Z = np.round(Z) if binary else Z
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.title(title + ' Decision boundary')
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()
