import numpy as np
import matplotlib.pyplot as plt
from nn_from_scratch.loadData import *
from softmax import *


def sgd(grad_function, cost_function, X_train, X_test, W, C_train, C_test, bias, batch_size, learning_rate, epoch_num,
        epsilon=0.1):
    """
    SGD algorithm implementation.
    """
    costs = []
    accuracy_train = []
    accuracy_test = []
    for epoch in range(epoch_num):
        # Epoch start
        # shuffle the data and indicators before batching
        shuffler = np.random.permutation(X_train.shape[1])
        X_shuffled = X_train.T[shuffler].T
        C_shuffled = C_train.T[shuffler].T
        m = X_train.shape[1]
        for i in range(int(m / batch_size)):
            # for each batch:
            X_batch = X_shuffled[:, i * batch_size:i * batch_size + batch_size]
            C_batch = C_shuffled[:, i * batch_size:i * batch_size + batch_size]
            grad_W, grad_b, _ = grad_function(X_batch, W, bias, C_batch)
            W = W - learning_rate * grad_W
            bias = bias - learning_rate * grad_b
        cost, _ = cost_function(X_shuffled, W, bias, C_shuffled)
        costs.append(cost)
        accuracy_train.append(success_percentage(X_shuffled, W, C_shuffled, bias))
        accuracy_test.append(success_percentage(X_test, W, C_test, bias))
        # Epoch end
    return W, costs, accuracy_train, accuracy_test


def plot_accuracy(accuracy_train, accuracy_test, epoches):
    plt.plot(range(epoches), accuracy_train)
    plt.plot(range(epoches), accuracy_test)
    plt.xlabel('Epoch')
    plt.ylabel('Success Percentage')
    plt.title('SGD Success % Per Epoch')
    plt.legend(['Success % - Train', 'Success % - Test'])
    plt.show()


def success_percentage(X, W, C, bias):
    probs = sigmoid(X, W, bias)
    labels_pred = np.argmax(probs, axis=1)
    labels_true = np.argmax(C.T, axis=1)
    accuracy_rate = sum(labels_pred == labels_true) / C.shape[1]
    return accuracy_rate * 100


if __name__ == '__main__':
    iter_num = 400
    learning_rate = 0.001
    batch_size = 60
    trainSetX, trainSetY, testSetX, testSetY, theta, bias = extract_sgd_data('PeaksData')
    W_train, costs_train, accuracy_train, accuracy_test = sgd(softmax_grad, softmax_regression, trainSetX, testSetX,
                                                              theta, trainSetY, testSetY, bias, batch_size,
                                                              learning_rate, iter_num)
    plot_accuracy(accuracy_train, accuracy_test, iter_num)
