import numpy as np
from section2.ResNet.residual_network import ResNet
from section2.activations import tanh, tanh_grad
import matplotlib.pyplot as plt

iter_num = 10


def jacobian_test_layer_X(nn: ResNet, X_0):
    """
    Direct jacobian transposed test for a single layer w.r.t to the data
    :param nn: the neural network object
    :param X_0: initial data (we init with random data)
    :return: shows a plot of the zero order vs. first order approximation
    """
    _, X_0_new = nn.forward_step(X_0, nn.weights[0], nn.biases[0])
    Ws_0 = nn.weights[1]
    b_0 = nn.biases[1].copy()
    _, X_1 = nn.res_forward_step(X_0_new, Ws_0, b_0)
    n, m = X_1.shape
    out_dimensions = b_0.shape[0]
    u = np.ones((out_dimensions, m))
    # u = np.random.rand(out_dimensions, m)
    d = np.random.rand(*X_0_new.shape)
    d = (1 / np.linalg.norm(d)) * d

    g_x = np.dot(X_1.T, u).item()
    _, _, _, JtU_X = nn.backward_hidden_layer([X_0_new, X_1], 2, u)

    zero_order = np.zeros(iter_num)
    first_order = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    for i, epsilon in enumerate(epsilons):
        X_diff = X_0_new.copy()
        X_diff += d * epsilon
        _, X_eps_forward = nn.res_forward_step(X_diff, Ws_0, b_0)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, u).item()
        d_flat = d.reshape(-1, 1)
        zero_order[i] = abs(gx_epsilon - g_x)
        first_order[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ JtU_X)
    draw_results(zero_order, first_order)


def jacobian_test_layer_W1(nn: ResNet, X_0):
    """
        Direct jacobian transposed test for a single layer w.r.t to the weights
        :param nn: the neural network object
        :param X_0: initial data (we init with random data)
        :return: shows a plot of the zero order vs. first order approximation
        """
    _, X_0_new = nn.forward_step(X_0, nn.weights[0], nn.biases[0])
    Ws_0 = nn.weights[1]
    b_0 = nn.biases[1].copy()
    _, X_1 = nn.res_forward_step(X_0_new, Ws_0, b_0)
    n, m = X_1.shape
    out_dimensions = b_0.shape[0]
    u = np.ones((out_dimensions, m))
    # u = np.random.rand(out_dimensions, m)
    d = np.random.rand(*Ws_0[0].shape)
    d = (1 / np.linalg.norm(d)) * d

    g_x = np.dot(X_1.T, u).item()
    JtU_W1, _, _, _ = nn.backward_hidden_layer([X_0_new, X_1], 1, u)

    zero_order = np.zeros(iter_num)
    first_order = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    for i, epsilon in enumerate(epsilons):
        W1_diff = Ws_0[0].copy()
        W1_diff += d * epsilon
        _, X_eps_forward = nn.res_forward_step(X_0_new, [W1_diff, Ws_0[1]], b_0)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, u).item()
        d_flat = d.reshape(-1, 1)
        JtU_W1_flat = JtU_W1.reshape(-1, 1)
        zero_order[i] = abs(gx_epsilon - g_x)
        first_order[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ JtU_W1_flat)
    draw_results(zero_order, first_order, 'W1')


def jacobian_test_layer_b(nn: ResNet, X_0):
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
    _, _, JtU_b, _ = nn.backward_hidden_layer([X_0, X_1], 1, u)

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
    nn = ResNet(8, 4, 1, tanh, tanh_grad)
    X = np.random.rand(8, 1)
    # jacobian_test_layer_X(nn, X)
    jacobian_test_layer_W1(nn, X)
    # jacobian_test_layer_b(nn, X)
