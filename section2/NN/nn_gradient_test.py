import numpy as np
from section2.NN.NN import NN
from section2.activations import tanh, tanh_grad
import matplotlib.pyplot as plt

num_of_iterations = 15


def nn_grad_test_W(nn: NN, X):
    """
    Test the gradient of the whole network w.r.t the weights
    :param nn: the neural network object
    :param X: the data to test with (we send random data)
    :return: shows a plot of the zero order vs. first order approximations
    """
    C = np.zeros((10, 1))
    C[0] = 1
    F_0, _, _, nonlinear_layers = nn.forward(X, C)
    weight_grads, biases_grads = nn.backpropagation(nonlinear_layers, C)
    grad_W = np.concatenate([w.flatten() for w in weight_grads])

    # define a random d for the test and align dimensions
    ds_w = [np.random.rand(w.shape[0], w.shape[1]) for w in nn.weights]
    ds_w = [d / np.linalg.norm(d) for d in ds_w]
    d_flat = np.concatenate([d.flatten() for d in ds_w])
    nn_weights_original = nn.weights.copy()

    zero_order = []
    first_order = []
    print('\nGradient test w.r.t bias results:')
    print('k\t\terror order 0\t\terror order 1')
    for k in range(num_of_iterations):
        epsilon = 0.5 ** k
        nn.weights = [nn.weights[i] + epsilon * ds_w[i] for i in range(len(nn.weights))]
        F_k, _, _, _ = nn.forward(X, C)
        F_1 = F_0 + epsilon * d_flat.T @ grad_W
        zero_order.append(np.linalg.norm(F_k - F_0))
        first_order.append(np.linalg.norm((F_k - F_1)))
        print(k, '\t', np.linalg.norm(F_k - F_0), '\t', np.linalg.norm(F_k - F_1))
        nn.weights = nn_weights_original
    draw_results(zero_order, first_order)


def nn_grad_test_b(nn: NN, X):
    """
    Test the gradient of the whole network w.r.t the biases
    :param nn: the neural network object
    :param X: the data to test with (we send random data)
    :return: shows a plot of the zero order vs. first order approximations
    """
    C = np.zeros((10, 1))
    C[0] = 1
    F_0, _, _, nonlinear_layers = nn.forward(X, C)
    _, biases_grads = nn.backpropagation(nonlinear_layers, C)
    grad_b = np.concatenate([b.flatten() for b in biases_grads])

    # define a random d for the test
    ds_b = [np.random.rand(b.shape[0], b.shape[1]) for b in nn.biases]
    ds_b = [d / np.linalg.norm(d) for d in ds_b]
    d_flat = np.concatenate([d.flatten() for d in ds_b])
    nn_biases_original = nn.biases.copy()

    zero_order = []
    first_order = []
    print('\nGradient test w.r.t bias results:')
    print('k\t\terror order 0\t\terror order 1')
    for k in range(num_of_iterations):
        epsilon = 0.5 ** k
        nn.biases = [nn.biases[i] + epsilon * ds_b[i] for i in range(len(nn.biases))]
        F_k, _, _, _ = nn.forward(X, C)
        F_1 = F_0 + epsilon * d_flat.T @ grad_b
        zero_order.append(np.linalg.norm(F_k - F_0))
        first_order.append(np.linalg.norm((F_k - F_1)))
        print(k, '\t', np.linalg.norm(F_k - F_0), '\t', np.linalg.norm(F_k - F_1))
        nn.biases = nn_biases_original
    draw_results(zero_order, first_order, "Biases")


def draw_results(y_0, y_1, result_for='Weights'):
    plt.semilogy([i for i in range(num_of_iterations)], y_0)
    plt.semilogy([i for i in range(num_of_iterations)], y_1)
    plt.legend(["Zero order approx", "First order approx"])
    plt.title("Successful {res_for} Gradient Test In Semilog Scale".format(res_for=result_for))
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()


if __name__ == '__main__':
    nn = NN([50, 10, 10, 10, 10], tanh, tanh_grad)
    X = np.random.rand(50, 1)
    nn_grad_test_W(nn, X)
    nn_grad_test_b(nn, X)
