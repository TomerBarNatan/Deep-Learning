import numpy as np
from section2.NN.NN import NN
from section2.activations import tanh, tanh_grad
import matplotlib.pyplot as plt

iter_num = 8


def jacobian_test_layer_X(nn: NN, X_0):
    """
    Direct jacobian transposed test for a single layer w.r.t to the data
    :param nn: the neural network object
    :param X_0: initial data (we init with random data)
    :return: shows a plot of the zero order vs. first order approximation
    """
    W_0 = nn.weights[0].copy()
    b_0 = nn.biases[0].copy()
    _, X_1 = nn.forward_step(X_0, W_0, b_0)
    n, m = X_1.shape
    out_dimensions = b_0.shape[0]
    u = np.random.rand(out_dimensions, m)
    d = np.random.rand(*X_0.shape)
    d = (1 / np.linalg.norm(d)) * d

    g_x = np.dot(X_1.T, u).item()
    _, _, JtU_X = nn.hidden_layer_grad(X_0, W_0, b_0, u)

    zero_order = np.zeros(iter_num)
    first_order = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    for i, epsilon in enumerate(epsilons):
        X_diff = X_0.copy()
        X_diff += d * epsilon
        _, X_eps_forward = nn.forward_step(X_diff, W_0, b_0)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, u).item()
        d_flat = d.reshape(-1, 1)
        zero_order[i] = abs(gx_epsilon - g_x)
        first_order[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ JtU_X)
    draw_results(zero_order, first_order)


def jacobian_test_layer_W(nn: NN, X_0):
    """
        Direct jacobian transposed test for a single layer w.r.t to the weights
        :param nn: the neural network object
        :param X_0: initial data (we init with random data)
        :return: shows a plot of the zero order vs. first order approximation
        """
    W_0 = nn.weights[0].copy()
    b_0 = nn.biases[0].copy()
    _, X_1 = nn.forward_step(X_0, W_0, b_0)
    n, m = X_1.shape
    out_dimensions = b_0.shape[0]
    u = np.random.rand(out_dimensions, m)
    d = np.random.rand(*W_0.shape)
    d = (1 / np.linalg.norm(d)) * d

    g_x = np.dot(X_1.T, u).item()
    JtU_W, _, _ = nn.hidden_layer_grad(X_0, W_0, b_0, u)

    zero_order = np.zeros(iter_num)
    first_order = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    for i, epsilon in enumerate(epsilons):
        W_diff = W_0.copy()
        W_diff += d * epsilon
        _, X_eps_forward = nn.forward_step(X_0, W_diff, b_0)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, u).item()
        d_flat = d.reshape(-1, 1)
        JtU_W_flat = JtU_W.reshape(-1, 1)
        zero_order[i] = abs(gx_epsilon - g_x)
        first_order[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ JtU_W_flat)
    draw_results(zero_order, first_order, 'weights')


def jacobian_test_layer_b(nn: NN, X_0):
    """
        Direct jacobian transposed test for a single layer w.r.t to biases
        :param nn: the neural network object
        :param X_0: initial data (we init with random data)
        :return: shows a plot of the zero order vs. first order approximation
        """
    W_0 = nn.weights[0].copy()
    b_0 = nn.biases[0].copy()
    _, X_1 = nn.forward_step(X_0, W_0, b_0)
    n, m = X_1.shape
    out_dimensions = b_0.shape[0]
    u = np.random.rand(out_dimensions, m)
    d = np.random.rand(*b_0.shape)
    d = (1 / np.linalg.norm(d)) * d

    g_x = np.dot(X_1.T, u).item()
    _, JtU_b, _ = nn.hidden_layer_grad(X_0, W_0, b_0, u)

    zero_order = np.zeros(iter_num)
    first_order = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    for i, epsilon in enumerate(epsilons):
        b_diff = b_0.copy()
        b_diff += d * epsilon
        _, X_eps_forward = nn.forward_step(X_0, W_0, b_diff)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, u).item()
        zero_order[i] = abs(gx_epsilon - g_x)
        first_order[i] = abs(gx_epsilon - g_x - epsilon * d.T @ JtU_b)
    draw_results(zero_order, first_order, 'bias')


def draw_results(zero_order, first_order, wrt='data'):
    plt.semilogy(np.arange(1, iter_num + 1, 1), zero_order)
    plt.semilogy(np.arange(1, iter_num + 1, 1), first_order)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title(f'Direct jacobian transposed test of a single layer w.r.t the {wrt}')
    plt.legend(("Zero order approximation", "First order approximation"))
    plt.show()


if __name__ == '__main__':
    nn = NN([3, 5, 7, 8], tanh, tanh_grad)
    X = np.random.rand(3, 1)
    jacobian_test_layer_b(nn, X)
    jacobian_test_layer_X(nn, X)
    jacobian_test_layer_W(nn, X)
