import numpy as np
from Assignment1.section2.ResNet.residual_network import ResNet
from Assignment1.section2.activations import tanh, tanh_grad
import matplotlib.pyplot as plt


def jacobian_test_layer_X(nn: ResNet, X_0):
    """
    Direct jacobian transposed test for a single layer w.r.t to the data
    :param nn: the neural network object
    :param X_0: initial data (we init with random data)
    :return: shows a plot of the zero order vs. first order approximation
    """
    iter_num = 6
    Ws_0 = nn.weights[1]
    b_0 = nn.biases[1].copy()
    _, X_1 = nn.res_forward_step(X_0, Ws_0, b_0)
    n, m = X_1.shape
    out_dimensions = b_0.shape[0]
    u = np.random.rand(out_dimensions, m)
    u = (1 / np.linalg.norm(u)) * u
    d = np.random.rand(*X_0.shape)
    d = (1 / np.linalg.norm(d)) * d

    g_x = np.dot(X_1.T, u).item()
    _, _, _, JtU_X = nn.res_hidden_layer_grad(X_1, Ws_0, b_0, u)

    zero_order = np.zeros(iter_num)
    first_order = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    print('\nGradient test w.r.t W results:')
    print('k\t\terror order 0\t\terror order 1')
    for i, epsilon in enumerate(epsilons):
        X_diff = X_0.copy()
        X_diff += d * epsilon
        _, X_eps_forward = nn.res_forward_step(X_diff, Ws_0, b_0)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, u).item()
        d_flat = d.reshape(-1, 1)
        zero_order[i] = abs(gx_epsilon - g_x)
        first_order[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ JtU_X)
        print(i, '\t', zero_order[i], '\t', first_order[i])
    draw_results(zero_order, first_order, iter_num)


def jacobian_test_layer_W(nn: ResNet, X_0):
    """
        Direct jacobian transposed test for a single layer w.r.t to the weights
        :param nn: the neural network object
        :param X_0: initial data (we init with random data)
        :return: shows a plot of the zero order vs. first order approximation
        """
    iter_num = 3
    _, X_1 = nn.forward_step(X_0, nn.weights[0], nn.biases[0])
    Ws_1 = nn.weights[1]
    b_1 = nn.biases[1].copy()
    _, X_2 = nn.res_forward_step(X_1, Ws_1, b_1)
    n, m = X_2.shape
    out_dimensions = b_1.shape[0]
    u = np.random.rand(out_dimensions, m)
    d1 = np.random.rand(*Ws_1[0].shape)
    d1 = (1 / np.linalg.norm(d1)) * d1
    d2 = np.random.rand(*Ws_1[1].shape)
    d2 = (1 / np.linalg.norm(d2)) * d2

    g_x = np.dot(X_1.T, u).item()
    JtU_W1, JtU_W2, _, _ = nn.res_hidden_layer_grad(X_1, Ws_1, b_1, u)

    zero_order = np.zeros(iter_num)
    first_order = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    print('\nGradient test w.r.t W results:')
    print('k\t\terror order 0\t\terror order 1')
    for i, epsilon in enumerate(epsilons):
        Ws_diff = Ws_1.copy()
        Ws_diff[0] = Ws_diff[0] + d1 * epsilon
        Ws_diff[1] = Ws_diff[1] + d2 * epsilon
        _, X_eps_forward = nn.res_forward_step(X_1, Ws_diff, b_1)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, u).item()
        d1_flat = d1.reshape(-1, 1)
        d2_flat = d2.reshape(-1, 1)
        JtU_W1_flat = JtU_W1.reshape(-1, 1)
        JtU_W2_flat = JtU_W2.reshape(-1, 1)
        zero_order[i] = abs(gx_epsilon - g_x)
        first_order[i] = abs(gx_epsilon - g_x - epsilon * d1_flat.T @ JtU_W1_flat - epsilon * d2_flat.T @ JtU_W2_flat)
        print(i, '\t', zero_order[i], '\t', first_order[i])
    draw_results(zero_order, first_order, iter_num, 'W')


def jacobian_test_layer_b(nn: ResNet, X_0):
    """
        Direct jacobian transposed test for a single layer w.r.t to biases
        :param nn: the neural network object
        :param X_0: initial data (we init with random data)
        :return: shows a plot of the zero order vs. first order approximation
        """
    iter_num = 8
    W_0 = nn.weights[1].copy()
    b_0 = nn.biases[1].copy()
    _, X_1 = nn.res_forward_step(X_0, W_0, b_0)
    n, m = X_1.shape
    out_dimensions = b_0.shape[0]
    u = np.random.rand(out_dimensions, m)
    d = np.random.rand(*b_0.shape)
    d = (1 / np.linalg.norm(d)) * d

    g_x = np.dot(X_1.T, u).item()
    _, _, JtU_b, _ = nn.backward_hidden_layer([X_0, X_1], 2, u)

    zero_order = np.zeros(iter_num)
    first_order = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    print('\nGradient test w.r.t W results:')
    print('k\t\terror order 0\t\terror order 1')
    for i, epsilon in enumerate(epsilons):
        b_diff = b_0.copy()
        b_diff += d * epsilon
        _, X_eps_forward = nn.res_forward_step(X_0, W_0, b_diff)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, u).item()
        zero_order[i] = abs(gx_epsilon - g_x)
        first_order[i] = abs(gx_epsilon - g_x - epsilon * d.T @ JtU_b)
        print(i, '\t', zero_order[i], '\t', first_order[i])
    draw_results(zero_order, first_order, iter_num, 'bias')


def draw_results(zero_order, first_order, iter_num, wrt='data'):
    plt.semilogy(np.arange(1, iter_num + 1, 1), zero_order)
    plt.semilogy(np.arange(1, iter_num + 1, 1), first_order)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title(f'Direct jacobian transposed test of a single layer w.r.t the {wrt}')
    plt.legend(("Zero order approximation", "First order approximation"))
    plt.show()


if __name__ == '__main__':
    nn = ResNet(2, 4, 1, tanh, tanh_grad, first_layer=2)
    X = np.random.rand(2, 1)
    jacobian_test_layer_X(nn, X)
    jacobian_test_layer_W(nn, X)
    jacobian_test_layer_b(nn, X)
