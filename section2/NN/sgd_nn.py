import matplotlib.pyplot as plt
from data.loadData import *
from section2.activations import *
from section1.softmax import *
from section2.NN.NN import NN


def sgd(nn: NN, X_train, X_test, W, C_train, C_test, batch_size, learning_rate, epoch_num, divide_lr=500,
        graph_till_now=None):
    """
    SGD for the neural network. each SGD batch iteration, the network learns using forward pass. then, backpropagation
    occurs to calculate the gradients and update the network params.
    :param nn: the neural network object
    :param X_train: train data
    :param X_test: test data
    :param W: weights
    :param C_train: train indicators
    :param C_test: test indicators
    :param batch_size: the desired batch size
    :param learning_rate: the desired learning rate
    :param epoch_num: number of epochs to perform
    :param divide_lr: how many epochs until dividing the learning rate by 10
    :return: Optimized weights, costs through the optimization, accuracy lists for the train and test
    """
    costs = []
    accuracy_train = []
    accuracy_test = []
    m = X_train.shape[1]
    for epoch in range(epoch_num):
        cur_costs = []
        if epoch % divide_lr == 0:
            learning_rate /= 10
        shuffler = np.random.permutation(X_train.shape[1])
        print(epoch)
        X_shuffled = X_train.T[shuffler].T
        C_shuffled = C_train.T[shuffler].T
        for i in range(int(m / batch_size)):
            X_batch = X_shuffled[:, i * batch_size:i * batch_size + batch_size]
            C_batch = C_shuffled[:, i * batch_size:i * batch_size + batch_size]
            cost, probs, linear_layers, nonlinear_layers = nn.forward(X_batch, C_batch)
            weights_grads, biases_grads = nn.backpropagation(nonlinear_layers, C_batch)
            nn.update_thetas(weights_grads, biases_grads, learning_rate)
            cur_costs.append(cost)
        costs.append(sum(cur_costs) / len(cur_costs))
        accuracy_train.append(success_percentage(nn, X_shuffled, C_shuffled))
        accuracy_test.append(success_percentage(nn, X_test, C_test))
        if graph_till_now and epoch % graph_till_now == 0 and epoch > 0:
            # plt.plot([i for i in range(epoch + 1)], costs)
            # plt.show()
            plot_accuracy(accuracy_train, accuracy_test, epoch + 1)
    return W, costs, accuracy_train, accuracy_test


def plot_accuracy(accuracy_train, accuracy_test, epochs):
    """
    Plots the accuracy graphs of the SGD iterations
    :param accuracy_train: train data accuracy
    :param accuracy_test: test data accuracy
    :param epochs: number of epochs performed
    """
    plt.plot(range(epochs), accuracy_train)
    plt.plot(range(epochs), accuracy_test)
    plt.xlabel('Epoch')
    plt.ylabel('Success Percentage')
    plt.title('SGD Success % Per Epoch')
    plt.legend(['Success % - Train', 'Success % - Test'])
    plt.show()


def success_percentage(nn: NN, X, C):
    """
    Calculates the success percentage of the optimization w.r.t the network
    :param nn: the neural network object
    :param X: the data
    :param C: the indicators
    :return: Accuracy percentage
    """
    _, probs, _, _ = nn.forward(X, C)
    labels_pred = np.argmax(probs, axis=0)
    labels_true = np.argmax(C.T, axis=1)
    accuracy_rate = sum(labels_pred == labels_true) / C.shape[1]
    return accuracy_rate * 100


def sgd_nn_peaks_data():
    """
    Run SGD with NN on Peaks data set
    """
    iter_num = 200
    learning_rate = 10
    batch_size = 60
    nn = NN([2, 5, 6, 8, 10, 5], ReLU, ReLU_grad)
    trainSetX, trainSetY, testSetX, testSetY, theta, bias = extract_sgd_data("PeaksData")
    W_train, costs_train, accuracy_train, accuracy_test = sgd(nn, trainSetX, testSetX, theta, trainSetY, testSetY,
                                                              batch_size, learning_rate, iter_num, graph_till_now=50)
    plot_accuracy(accuracy_train, accuracy_test, iter_num)


def sgd_nn_gmm_data():
    """
    Run SGD with NN on GMM data set
    """
    iter_num = 200
    learning_rate = 100
    batch_size = 60
    nn = NN([5, 5, 6, 8, 10, 5], tanh, tanh_grad)
    trainSetX, trainSetY, testSetX, testSetY, theta, bias = extract_sgd_data("GMMData")
    W_train, costs_train, accuracy_train, accuracy_test = sgd(nn, trainSetX, testSetX, theta, trainSetY, testSetY,
                                                              batch_size, learning_rate, iter_num)
    plot_accuracy(accuracy_train, accuracy_test, iter_num)


def sgd_nn_swiss_roll_data():
    """
    Run SGD with NN on SwissRoll data set
    """
    iter_num = 1000
    learning_rate = 0.3
    batch_size = 17
    nn = NN([2, 5, 6, 6, 4, 5, 2], ReLU, ReLU_grad)
    trainSetX, trainSetY, testSetX, testSetY, theta, bias = extract_sgd_data("SwissRollData")
    W_train, costs_train, accuracy_train, accuracy_test = sgd(nn, trainSetX, testSetX, theta, trainSetY, testSetY,
                                                              batch_size, learning_rate, iter_num, 10000, 50)
    plot_accuracy(accuracy_train, accuracy_test, iter_num)



def sgd_nn_small_train_peaks_data():
    """
    Run SGD with NN on Peaks data set
    """
    iter_num = 1000
    learning_rate = 1
    batch_size = 5
    nn = NN([2, 5, 6, 5], tanh, tanh_grad)
    trainSetX, trainSetY, testSetX, testSetY, theta, bias = extract_sgd_data("../../PeaksData")

    shuffler = np.random.permutation(trainSetX.shape[1])
    X_shuffled = trainSetX.T[shuffler].T
    C_shuffled = trainSetY.T[shuffler].T
    W_train, costs_train, accuracy_train, accuracy_test = sgd(nn, X_shuffled[:,:200], testSetX, theta, C_shuffled[:,:200], testSetY,
                                                              batch_size, learning_rate, iter_num, graph_till_now=200)
    plot_accuracy(accuracy_train, accuracy_test, iter_num)


if __name__ == '__main__':
    sgd_nn_peaks_data()
