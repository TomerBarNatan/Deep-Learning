import matplotlib.pyplot as plt
from loadData import *
from nn_from_scratch.section2.ResNet.residual_network import ResNet
from nn_from_scratch.section2.activations import *


def sgd(nn: ResNet, X_train, X_test, W, C_train, C_test, batch_size, learning_rate, iter_num, divide_lr=50, graph_till_now = None):
    costs = []
    accuracy_train = []
    accuracy_test = []
    m = X_train.shape[1]
    for iter in range(iter_num):
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
            costs.append(cost)
        accuracy_train.append(success_percentage(nn, X_shuffled, C_shuffled))
        accuracy_test.append(success_percentage(nn, X_test, C_test))
        if graph_till_now and iter % graph_till_now == 0 and iter > 0:
            plot_accuracy(accuracy_train, accuracy_test, iter+1)
    return W, costs, accuracy_train, accuracy_test


def plot_accuracy(accuracy_train, accuracy_test, epoches):
    plt.plot(range(epoches), accuracy_train)
    plt.plot(range(epoches), accuracy_test)
    plt.xlabel('Epoch')
    plt.ylabel('Success Percentage')
    plt.title('SGD Success % Per Epoch')
    plt.legend(['Success % - Train', 'Success % - Test'])
    plt.show()


def success_percentage(nn: ResNet, X, C):
    _, probs, _, _ = nn.forward(X, C)
    labels_pred = np.argmax(probs, axis=0)
    labels_true = np.argmax(C.T, axis=1)
    accuracy_rate = sum(labels_pred == labels_true) / C.shape[1]
    return accuracy_rate * 100


def sgd_nn_peaks_data():
    iter_num = 1000
    learning_rate = 100
    batch_size = 64
    rn = ResNet(2,4,5, ReLU, ReLU_grad, first_layer= 8)
    trainSetX, trainSetY, testSetX, testSetY, theta, bias = extract_sgd_data("PeaksData")
    W_train, costs_train, accuracy_train, accuracy_test = sgd(rn, trainSetX, testSetX, theta, trainSetY, testSetY,
                                                              batch_size, learning_rate, iter_num,divide_lr=100,graph_till_now=50 )
    plot_accuracy(accuracy_train, accuracy_test, iter_num)


def sgd_nn_swiss_roll_data():
    iter_num = 1000
    learning_rate = 10
    batch_size = 20000
    rn = ResNet(2,6,2, ReLU, ReLU_grad, first_layer= 32)
    trainSetX, trainSetY, testSetX, testSetY, theta, bias = extract_sgd_data("SwissRollData")
    W_train, costs_train, accuracy_train, accuracy_test = sgd(rn, trainSetX, testSetX, theta, trainSetY, testSetY,
                                                              batch_size, learning_rate, iter_num,divide_lr=200,graph_till_now=100 )
    plot_accuracy(accuracy_train, accuracy_test, iter_num)


if __name__ == '__main__':
    sgd_nn_peaks_data()
