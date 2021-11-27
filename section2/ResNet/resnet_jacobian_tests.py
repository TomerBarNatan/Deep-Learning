import numpy as np
from section2.ResNet.residual_network import ResNet
from section2.activations import tanh, tanh_grad
import matplotlib.pyplot as plt

iter_num = 10


def jacobian_test_layer_X(nn: ResNet, X_0):
    """
    Direct jacobian transposed test for a single layer w.r.t to the data
    :param nn: the residual neural network object
    :param X_0: initial data (we init with random data)
    :return: shows a plot of the zero order vs. first order approximation
    """
    _, X_1 = nn.forward_step(X_0, nn.weights[0], nn.biases[0])
    _, X = nn.res_forward_step(X_1, nn.weights[1], nn.biases[1])
    n, m = X.shape
    out_dimensions = nn.biases[2].shape[0]
    U = np.random.rand(out_dimensions, m)
    zero_order = np.zeros(iter_num)
    first_order = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    d = np.ones((X.shape[0], X.shape[1]))
    d /= np.linalg.norm(d)
    W = nn.weights[2]
    b = nn.biases[2]

    X_linear, X_forward = nn.res_forward_step(X, W, b)
    X_forward_T = X_forward.T
    g_x = np.dot(X_forward_T, U).item()

    temp = W[1] @ np.diag(nn.activation_gradient(X_linear).flatten()) @ W[0]
    Jx = np.eye(temp.shape[0]) + temp
    g_grad = Jx.T @ U
    for i, epsilon in enumerate(epsilons):
        X_diff = X.copy()
        X_diff += d * epsilon
        _, X_eps_forward = nn.res_forward_step(X_diff, W, b)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, U).item()
        d_flat = d.reshape(-1, 1)
        zero_order[i] = abs(gx_epsilon - g_x)
        first_order[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ g_grad)
    draw_results(zero_order, first_order)


def jacobian_test_layer_W1(nn: ResNet, X_0):
    """
        Direct jacobian transposed test for a single layer w.r.t to W1
        :param nn: the residual neural network object
        :param X_0: initial data (we init with random data)
        :return: shows a plot of the zero order vs. first order approximation
        """
    _, X_1 = nn.forward_step(X_0, nn.weights[0], nn.biases[0])
    _, X = nn.res_forward_step(X_1, nn.weights[1], nn.biases[1])
    n, m = X.shape
    out_dimensions = nn.biases[2].shape[0]
    U = np.random.rand(out_dimensions, m)
    zero_order = np.zeros(iter_num)
    first_order = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    W = nn.weights[2]
    b = nn.biases[2]

    d = np.ones((W[0].shape[0], W[0].shape[1]))
    d /= np.linalg.norm(d)

    X_linear, X_forward = nn.res_forward_step(X, W, b)
    X_forward_T = X_forward.T
    g_x = np.dot(X_forward_T, U).item()

    D = np.diag(nn.activation_gradient(X_linear).flatten())
    Jw1 = W[1] @ D @ np.kron(X.T, np.eye(int(D.shape[1] / X.T.shape[0])))
    g_grad = Jw1.T @ U

    for i, epsilon in enumerate(epsilons):
        W_diff = W.copy()
        W_diff[0] = W_diff[0] + d * epsilon
        _, X_eps_forward = nn.res_forward_step(X, W_diff, b)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, U).item()
        d_flat = d.reshape(-1, 1)
        zero_order[i] = abs(gx_epsilon - g_x)
        first_order[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ g_grad)
    draw_results(zero_order, first_order, "W1")


def jacobian_test_layer_W2(nn: ResNet, X_0):
    """
        Direct jacobian transposed test for a single layer w.r.t to W1
        :param nn: the residual neural network object
        :param X_0: initial data (we init with random data)
        :return: shows a plot of the zero order vs. first order approximation
        """
    _, X_1 = nn.forward_step(X_0, nn.weights[0], nn.biases[0])
    _, X = nn.res_forward_step(X_1, nn.weights[1], nn.biases[1])
    n, m = X.shape
    out_dimensions = nn.biases[2].shape[0]
    U = np.random.rand(out_dimensions, m)
    zero_order = np.zeros(iter_num)
    first_order = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    W = nn.weights[2]
    b = nn.biases[2]

    d = np.random.rand(W[1].shape[0], W[1].shape[1])
    d /= np.linalg.norm(d)

    X_linear, X_forward = nn.res_forward_step(X, W, b)
    X_forward_T = X_forward.T
    g_x = np.dot(X_forward_T, U).item()
    temp = nn.activation(X_linear).T
    Jw2 = np.kron(temp, np.eye(temp.shape[1]))
    g_grad = Jw2.T @ U

    for i, epsilon in enumerate(epsilons):
        W_diff = W.copy()
        W_diff[1] = W_diff[1] + d * epsilon
        _, X_eps_forward = nn.res_forward_step(X, W_diff, b)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, U).item()
        d_flat = d.reshape(-1, 1)
        zero_order[i] = abs(gx_epsilon - g_x)
        first_order[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ g_grad)
    draw_results(zero_order, first_order, "W2")


def jacobian_test_layer_b(nn: ResNet, X_0):
    """
        Direct jacobian transposed test for a single layer w.r.t to biases
        :param nn: the residual neural network object
        :param X_0: initial data (we init with random data)
        :return: shows a plot of the zero order vs. first order approximation
        """
    _, X_1 = nn.forward_step(X_0, nn.weights[0], nn.biases[0])
    _, X = nn.res_forward_step(X_1, nn.weights[1], nn.biases[1])
    n, m = X.shape
    out_dimensions = nn.biases[2].shape[0]
    U = np.random.rand(out_dimensions, m)
    zero_order = np.zeros(iter_num)
    first_order = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    W = nn.weights[2]
    b = nn.biases[2]

    d = np.ones((b.shape[0], b.shape[1]))
    d /= np.linalg.norm(d)

    X_linear, X_forward = nn.res_forward_step(X, W, b)
    X_forward_T = X_forward.T
    g_x = np.dot(X_forward_T, U).item()

    D = np.diag(nn.activation_gradient(X_linear).flatten())
    Jb = W[1] @ D
    g_grad = Jb.T @ U

    for i, epsilon in enumerate(epsilons):
        b_diff = b.copy()
        b_diff = b_diff + d * epsilon
        _, X_eps_forward = nn.res_forward_step(X, W, b_diff)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, U).item()
        d_flat = d.reshape(-1, 1)
        zero_order[i] = abs(gx_epsilon - g_x)
        first_order[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ g_grad)
    draw_results(zero_order, first_order, "bias")


def draw_results(zero_order, first_order, wrt='data'):
    plt.semilogy(np.arange(1, iter_num + 1, 1), zero_order)
    plt.semilogy(np.arange(1, iter_num + 1, 1), first_order)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title(f'Direct jacobian transposed test of a single layer w.r.t the {wrt}')
    plt.legend(("Zero order approximation", "First order approximation"))
    plt.show()


if __name__ == '__main__':
    nn = ResNet(3, 5, 10, tanh, tanh_grad)
    X = np.random.rand(3, 1)
    # jacobian_test_layer_b(nn, X)
    # jacobian_test_layer_X(nn, X)
    # jacobian_test_layer_W1(nn, X)
    jacobian_test_layer_W2(nn, X)
