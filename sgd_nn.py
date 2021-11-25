import numpy as np
import matplotlib.pyplot as plt
from loadData import *
from softmax import *
from NN import NN


def sgd(nn: NN, X_train, X_test, W, C_train, C_test, batch_size, learning_rate, iter_num, divide_lr=50,
        graph_till_now=None):
    costs = []
    accuracy_train = []
    accuracy_test = []
    m = X_train.shape[1]
    for iter in range(iter_num):
        cur_costs = []
        if iter % divide_lr == 0:
            learning_rate /= 10
        shuffler = np.random.permutation(X_train.shape[1])
        print(iter)
        X_shuffled = X_train.T[shuffler].T
        C_shuffled = C_train.T[shuffler].T
        for i in range(int(m / batch_size)):
            X_batch = X_shuffled[:, i * batch_size:i * batch_size + batch_size]
            C_batch = C_shuffled[:, i * batch_size:i * batch_size + batch_size]
            cost, probs, linear_layers, nonlinear_layers = nn.forward(X_batch, C_batch)
            weight_grads, bias_grads = nn.backpropagation(nonlinear_layers, C_batch)
            nn.update_thetas(weight_grads, bias_grads, learning_rate)
            cur_costs.append(cost)
        costs.append(sum(cur_costs) / len(cur_costs))
        accuracy_train.append(success_percentage(nn, X_shuffled, C_shuffled))
        accuracy_test.append(success_percentage(nn, X_test, C_test))
        if graph_till_now and iter % graph_till_now == 0 and iter > 0:
            # plt.plot([i for i in range(iter + 1)], costs)
            # plt.show()
            plot_accuracy(accuracy_train, accuracy_test, iter + 1)
    return W, costs, accuracy_train, accuracy_test


def plot_accuracy(accuracy_train, accuracy_test, epoches):
    plt.plot(range(epoches), accuracy_train)
    plt.plot(range(epoches), accuracy_test)
    plt.xlabel('Epoch')
    plt.ylabel('Success Percentage')
    plt.title('SGD Success % Per Epoch')
    plt.legend(['Success % - Train', 'Success % - Test'])
    plt.show()


def success_percentage(nn: NN, X, C):
    _, probs, _, _ = nn.forward(X, C)
    labels_pred = np.argmax(probs, axis=0)
    labels_true = np.argmax(C.T, axis=1)
    accuracy_rate = sum(labels_pred == labels_true) / C.shape[1]
    return accuracy_rate * 100


def tan_h_gradient(x):
    return np.ones(x.shape) - (np.tanh(x)) ** 2


def sgd_nn_peaks_data():
    iter_num = 200
    learning_rate = 10
    batch_size = 60
    nn = NN([2, 5, 6, 8, 10, 5], relu, relu_grad)
    trainSetX, trainSetY, testSetX, testSetY, theta, bias = extract_sgd_data("PeaksData")
    W_train, costs_train, accuracy_train, accuracy_test = sgd(nn, trainSetX, testSetX, theta, trainSetY, testSetY,
                                                              batch_size, learning_rate, iter_num, graph_till_now=50)
    plot_accuracy(accuracy_train, accuracy_test, iter_num)


def sgd_nn_gmm_data():
    iter_num = 200
    learning_rate = 100
    batch_size = 60
    nn = NN([5, 5, 6, 8, 10, 5], np.tanh, tan_h_gradient)
    trainSetX, trainSetY, testSetX, testSetY, theta, bias = extract_sgd_data("GMMData")
    W_train, costs_train, accuracy_train, accuracy_test = sgd(nn, trainSetX, testSetX, theta, trainSetY, testSetY,
                                                              batch_size, learning_rate, iter_num)
    plot_accuracy(accuracy_train, accuracy_test, iter_num)


def relu(x):
    return np.maximum(x, 0)


def relu_grad(x):
    f = lambda t: 1 if t >= 0 else 0
    vfunc = np.vectorize(f)
    return vfunc(x)


def sgd_nn_swiss_roll_data():
    iter_num = 1000
    learning_rate = 0.3
    batch_size = 17
    nn = NN([2, 5, 6, 6, 4, 5, 2], relu, relu_grad)
    trainSetX, trainSetY, testSetX, testSetY, theta, bias = extract_sgd_data("SwissRollData")
    W_train, costs_train, accuracy_train, accuracy_test = sgd(nn, trainSetX, testSetX, theta, trainSetY, testSetY,
                                                              batch_size, learning_rate, iter_num, 10000, 50)
    plot_accuracy(accuracy_train, accuracy_test, iter_num)


if __name__ == '__main__':
    sgd_nn_peaks_data()
