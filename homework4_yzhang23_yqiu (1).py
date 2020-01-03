#author Ying Zhang & Yujia Qiu
# CS 453X
# Professor Jacob Whitehill
# Hw5

import numpy as np
import matplotlib.pyplot as plt

def PCA(X):
    AvgX = np.mean(X,axis = 1)
    listX = np.array([AvgX]).T
    side1 = X -listX
    s1_idx, s2_idx = np.linalg.eigh(np.dot(side1, side1.T))
    p1_idx = -1
    p2_idx = -2
    p1 = s2_idx[:, p1_idx]
    p2 = s2_idx[:, p2_idx]
    return p1, p2


if __name__ == "__main__":
    X = np.load('small_mnist_test_images.npy').T
    #y = np.load('small_mnist_test_labels.npy').idxmax(axis = 1).val
    y = np.load("small_mnist_test_labels.npy")
    p1,p2 = PCA(X)
    c = ['tomato', 'peru', 'paleturquoise', 'darkslategray', 'teal', 'aqua', 'deepskyblue', 'steelblue','dodgerblue','seagreen','darkgreen','lime']
    for i in range(10):
        index = np.nonzero(y[:, i] == 1)[0]
        p1s = np.dot(X[:, index].T, p1)
        p2s = np.dot(X[:, index].T, p2)
        #plt.scatter(p1, p2, c = y, s = 1)
        #c' argument has 5000 elements, which is not acceptable for use with 'x' with size 460, 'y' with size 460.
        plt.scatter(p1s, p2s , c = c[i], s = 5)
    plt.title('PCA')
    plt.show()
