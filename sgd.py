import numpy as np
import matplotlib.pyplot as plt
from loadData import *
from softmax import *


def sgd(grad_function, cost_function, X, W, C, bias,batch_size, learning_rate, iter_num, epsilon = 0.1):
    costs = []
    for iter in range(iter_num):
        shuffler = np.random.permutation(X.shape[1])
        X = X.T[shuffler]
        C = C.T[shuffler].T
        for i in range(batch_size):
            X_i = X[i * batch_size:i * batch_size + batch_size, :]
            C_i = C[i * batch_size:i * batch_size + batch_size]
            grad = grad_function(X_i, W, C_i, bias)
            W = W - (1 / batch_size) * learning_rate * (grad)
        costs.append(cost_function(X, W, C, bias))
    return W, costs


def least_squares_gradient(A, x, b, bias = None):
    grad = 2 * A.T @ A @ x - 2 * A.T @ b
    return grad


def least_squares_cost_function(A, x, b, bias = None):
    cost = np.linalg.norm(A @ x - b)
    return cost


if __name__ == '__main__':
    # iter_num = 1000
    # # A = np.array([[-1, -1, 1], [1, 3, 3], [-1, -1, 5], [1, 3, 7], [1, 2, 3], [1, 3, 4]])
    # # b = np.array([[0, 23, 15, 39, 1, 12]]).T
    # A = np.random.rand(3, 2)
    # b = np.random.rand(3, 1)
    # x = np.random.rand(2, 1)
    # x, costs = sgd(least_squares_gradient, least_squares_cost_function, A, x, b, None,10, 0.5, iter_num)
    # actual = np.linalg.lstsq(A, b)
    # # errors = [np.linalg.norm(W - actual[0]) for W in costs]
    # plt.semilogy(range(iter_num), costs)
    # print(A@x - b)
    # plt.show()

    trainSetX, trainSetY, theta, bias = extract_sgd_data()
    W, costs = sgd(softmax_grad, softmax_regression, trainSetX, theta, trainSetY, bias, 100, 0.001, 1000)