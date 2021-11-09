import numpy as np
import matplotlib.pyplot as plt
from loadData import *
from softmax import *


def sgd(grad_function, cost_function, X_train, X_test, W, C_train, C_test, bias, batch_size, learning_rate, iter_num, epsilon = 0.1):
    costs = []
    accuracy_train = []
    accuracy_test = []
    for iter in range(iter_num):
        shuffler = np.random.permutation(X_train.shape[1])
        X_shuffled = X_train.T[shuffler].T
        C_shuffled = C_train.T[shuffler].T
        m = X_train.shape[1]
        for i in range(int(m / batch_size)):
            X_batch = X_shuffled[:, i * batch_size:i * batch_size + batch_size]
            C_batch = C_shuffled[:, i * batch_size:i * batch_size + batch_size]
            grad_W, grad_b = grad_function(X_batch, W, bias, C_batch)
            W = W - learning_rate * grad_W
            bias = bias - learning_rate * grad_b
        costs.append(cost_function(X_shuffled, W, bias, C_shuffled))
        accuracy_train.append(accuracy_percentage(X_shuffled, W, C_shuffled, bias))
        accuracy_test.append(accuracy_percentage(X_test, W, C_test, bias))
    return W, costs, accuracy_train, accuracy_test


def least_squares_gradient(A, x, bias, b):
    grad = 2 * A.T @ A @ x - 2 * A.T @ b
    return grad


def least_squares_cost_function(A, x, b, bias = None):
    cost = np.linalg.norm(A @ x - b)
    return cost


def plot_results(costs, iter_num):
    plt.semilogy(range(iter_num), costs)
    plt.show()


def plot_accuracy(accuracy_train, accuracy_test, epoches):
    plt.plot(range(epoches), accuracy_train)
    plt.plot(range(epoches), accuracy_test)
    plt.show()


def accuracy_percentage(X, W, C, bias):
    probs = sigmoid(X, W, bias)
    labels_pred = np.argmax(probs, axis=1)
    labels_true = np.argmax(C.T, axis=1)
    accuracy_rate = sum(labels_pred == labels_true) / C.shape[1]
    return accuracy_rate * 100



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

    iter_num = 400
    trainSetX, trainSetY, testSetX, testSetY, theta, bias = extract_sgd_data()
    W_train, costs_train, accuracy_train, accuracy_test = sgd(softmax_grad, softmax_regression, trainSetX, testSetX, theta, trainSetY, testSetY, bias, 62, 0.001, iter_num)
    plot_accuracy(accuracy_train, accuracy_test, iter_num)
    # plot_results(costs, iter_num)