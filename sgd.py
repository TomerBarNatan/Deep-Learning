import numpy as np
import matplotlib.pyplot as plt
from loadData import extract_data
from softmax import *


def sgd(grad_function ,X, W, C, batch_size, learning_rate, iter_num):
    costs = []
    for iter in range(iter_num):
        shuffler = np.random.permutation(len(C))
        X = X[shuffler]
        C = C[shuffler]
        for i in range(batch_size):
            X_i = X[i*batch_size:i*batch_size+batch_size, :]
            C_i = C[i*batch_size:i*batch_size+batch_size]
            grad = grad_function(X_i, W, C_i)
            W -= (1 / batch_size) * learning_rate * (grad)
        costs.append(W)
    return W, costs


def least_squares_gradient(A, x, b):
    grad = 2*A.T @ A @ x -2*A.T @ b
    return grad


if __name__ == '__main__':
    iter_num = 8000
    A = np.array([[-1, -1, 1], [1, 3, 3], [-1, -1, 5], [1, 3, 7]])
    b = np.array([[0, 23, 15, 39]]).T
    x = np.zeros((3,1))
    x, costs = sgd(least_squares_gradient, A, x, b, 4, 0.01, iter_num)
    actual = np.linalg.lstsq(A,b)
    errors = [np.linalg.norm(W-actual[0]) for W in costs]
    plt.semilogy(range(iter_num), costs)
    plt.show()