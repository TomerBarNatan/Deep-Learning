import matplotlib.pyplot as plt
from Assignment1.data.loadData import *
from Assignment1.section2.activations import *
from Assignment1.section1.softmax import *
from Assignment1.section2.NN.NN import NN


def sgd(data_set, nn: NN, batch_size, learning_rate, epoch_num, train_size=None, divide_lr=50, graph_till_now=None,
        should_plot=True):
    """
    SGD for the neural network. each SGD batch iteration, the network learns using forward pass. then, backpropagation
    occurs to calculate the gradients and update the network params.
    :param data_set: the name of the data set to run
    :param nn: the neural network object
    :param batch_size: the desired batch size
    :param learning_rate: the desired learning rate
    :param epoch_num: number of epochs to perform
    :param train_size: size of train data to load
    :param divide_lr: how many epochs until dividing the learning rate by 10
    :param graph_till_now: show progress plot after some iterations
    :param should_plot: plot accuracy and cost at the end of the optimization
    :return: Optimized weights, costs through the optimization, accuracy lists for the train and test
    """
    X_train, C_train, X_test, C_test = extract_nn_sgd_data(data_set, train_size)
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
            _, weights_grads, biases_grads = nn.backpropagation(nonlinear_layers, C_batch)
            nn.update_thetas(weights_grads, biases_grads, learning_rate)
            cur_costs.append(cost)
        costs.append(sum(cur_costs) / len(cur_costs))
        accuracy_train.append(success_percentage(nn, X_shuffled, C_shuffled))
        accuracy_test.append(success_percentage(nn, X_test, C_test))
        # plot the accuracy progress
        if graph_till_now and epoch % graph_till_now == 0 and epoch > 0:
            plot_accuracy(accuracy_train, accuracy_test, epoch + 1)
    if should_plot:
        plot_accuracy(accuracy_train, accuracy_test, epoch_num, data_set)
        plot_cost(costs, epoch_num)
    return costs, accuracy_train, accuracy_test


def plot_accuracy(accuracy_train, accuracy_test, epochs, data_set):
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
    plt.title(f'SGD Success % Per Epoch for {data_set}')
    plt.legend(['Success % - Train', 'Success % - Test'])
    plt.show()


def plot_cost(costs, epochs):
    """
        Plots the cost graph of the SGD optimization
        :param costs: costs received in the optimization process
        :param epochs: number of epochs performed
        """
    plt.plot(range(epochs), costs)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Cost Function Per Epoch')
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
    iter_num = 400
    learning_rate = 10
    batch_size = 5
    nn = NN([2, 5, 6, 8, 10, 5], ReLU, ReLU_grad)
    sgd("PeaksData", nn, batch_size, learning_rate, iter_num,train_size=200)


def sgd_nn_peaks_data_multi_layer(train_size=None):
    """
    Run SGD with NN on Peaks data set
    """
    iter_num = 200
    batch_size = 5
    layers = [[2, 5], [2, 5, 5], [2, 5, 10, 5],[2, 5, 6, 8, 10, 5], [2, 5, 6, 8, 10, 16, 10, 5]]
    legend = []

    for layer in layers:
        learning_rate = 10
        nn = NN(layer, ReLU, ReLU_grad)
        costs_train, accuracy_train, accuracy_test = sgd("PeaksData", nn, batch_size, learning_rate, iter_num,
                                                         train_size=train_size, should_plot=False)
        # plt.plot(range(iter_num), accuracy_train)
        plt.plot(range(iter_num), accuracy_test)
        plt.xlabel('Epoch')
        plt.ylabel('Success Percentage')
        legend += [f'Success % - {len(layer)} layers']
    plt.legend(legend)
    plt.title('SGD Success % Per Epoch for PeaksData')
    plt.show()


def sgd_nn_gmm_data():
    """
    Run SGD with NN on GMM data set
    """
    iter_num = 400
    learning_rate = 1
    batch_size = 5
    nn = NN([5, 5, 6, 8, 10, 5], tanh, tanh_grad)
    sgd("GMMData", nn, batch_size, learning_rate, iter_num,divide_lr=300,train_size=200)


def sgd_nn_gmm_data_multi_layer(train_size=None):
    """
    Run SGD with NN on Peaks data set
    """
    iter_num = 400
    batch_size = 5
    layers = [[5, 5], [5, 10, 5], [5, 8, 10, 5],[5, 6, 10, 10, 5], [5, 10, 16, 10, 5]]
    legend = []
    for layer in layers:
        learning_rate = 1
        nn = NN(layer, ReLU, ReLU_grad)
        costs_train, accuracy_train, accuracy_test = sgd("GMMData", nn, batch_size, learning_rate, iter_num,
                                                         train_size=train_size, should_plot=False, divide_lr=300)
        # plt.plot(range(iter_num), accuracy_train)
        plt.plot(range(iter_num), accuracy_test)
        plt.xlabel('Epoch')
        plt.ylabel('Success Percentage')
    legend = [f'Success % - 2 layers',f'Success % - 4 layers',f'Success % - 4 layers',f'Success % - 6 layers',f'Success % - 8 layers']
    plt.legend(legend)
    plt.title('SGD Success % Per Epoch for GMMData')
    plt.show()


def sgd_nn_swiss_roll_data():
    """
    Run SGD with NN on SwissRoll data set
    """
    iter_num = 100
    learning_rate = 10
    batch_size = 5
    nn = NN([2, 5, 6, 6, 4, 2], ReLU, ReLU_grad)
    sgd("SwissRollData", nn, batch_size, learning_rate, iter_num, divide_lr=5000,train_size=200)


def sgd_nn_swiss_roll_data_multi_layer(train_size=None):
    """
    Run SGD with NN on Peaks data set
    """
    iter_num = 400
    batch_size = 17
    layers = [[2, 2], [2, 10, 2], [2, 5, 8, 2],[2, 5, 6, 6, 4, 2], [2, 6, 6, 8, 10, 16, 10, 2]]
    legend = []
    for layer in layers:
        learning_rate = 0.03
        nn = NN(layer, ReLU, ReLU_grad)
        costs_train, accuracy_train, accuracy_test = sgd("SwissRollData", nn, batch_size, learning_rate, iter_num,
                                                         train_size=train_size, should_plot=False)
        # plt.plot(range(iter_num), accuracy_train)
        plt.plot(range(iter_num), accuracy_test)
        plt.xlabel('Epoch')
        plt.ylabel('Success Percentage')
        legend += [f'Success % - {len(layer)} layers']
    plt.legend(legend)
    plt.title('SGD Success % Per Epoch for SwissRollData')
    plt.show()


if __name__ == '__main__':
    sgd_nn_peaks_data()
    sgd_nn_peaks_data_multi_layer()
    sgd_nn_peaks_data_multi_layer(train_size=200)

    sgd_nn_gmm_data()
    sgd_nn_gmm_data_multi_layer()
    sgd_nn_gmm_data_multi_layer(train_size=200)

    sgd_nn_swiss_roll_data()
    sgd_nn_swiss_roll_data_multi_layer()
    sgd_nn_swiss_roll_data_multi_layer(train_size=200)
