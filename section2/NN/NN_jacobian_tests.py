import numpy as np
from section2.NN.NN import NN
from section2.activations import tanh, tanh_grad
import matplotlib.pyplot as plt


def jacobian_test_layer_X(nn: NN, X_0):
    iter_num = 10
    _, X = nn.forward_step(X_0, nn.weights[0], nn.biases[0])
    n, m = X.shape
    out_dimensions = nn.biases[1].shape[0]
    U = np.random.rand(out_dimensions, m)
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    d = np.ones((X.shape[0], X.shape[1]))
    d /= np.linalg.norm(d)
    W = nn.weights[1]
    b = nn.biases[1]

    X_linear, X_forward = nn.forward_step(X, W, b)
    X_forward_T = X_forward.T
    g_x = np.dot(X_forward_T, U).item()

    Jx = np.diag(nn.activation_gradient(X_linear).flatten()) @ W
    g_grad = Jx.T @ U
    for i, epsilon in enumerate(epsilons):
        X_diff = X.copy()
        X_diff += d * epsilon
        _, X_eps_forward = nn.forward_step(X_diff, W, b)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, U).item()
        d_flat = d.reshape(-1, 1)
        diff[i] = abs(gx_epsilon - g_x)
        diff_grad[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ g_grad)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('X Jacobian Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()


def jacobian_test_layer_W(nn: NN, X_0):
    iter_num = 10
    _, X = nn.forward_step(X_0, nn.weights[0], nn.biases[0])
    W = nn.weights[1]
    b = nn.biases[1]
    n, m = X.shape
    out_dimensions = nn.biases[1].shape[0]
    U = np.random.rand(out_dimensions, m)
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    d = np.ones((W.shape[0], W.shape[1]))
    # d = np.random.rand(*W.shape)
    d /= np.linalg.norm(d)

    X_linear, X_forward = nn.forward_step(X, W, b)
    X_forward_T = X_forward.T
    g_x = np.dot(X_forward_T, U).item()

    D = np.diag(nn.activation_gradient(X_linear).flatten())
    Jx = D @ np.kron(X.T, np.eye(int(D.shape[1] / X.T.shape[0])))
    g_grad = Jx.T @ U
    for i, epsilon in enumerate(epsilons):
        W_diff = W.copy()
        W_diff += d * epsilon
        _, X_eps_forward = nn.forward_step(X, W_diff, b)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, U).item()
        d_flat = d.reshape(-1, 1)
        diff[i] = abs(gx_epsilon - g_x)
        diff_grad[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ g_grad)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('W Jacobian Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()


def jacobian_test_layer_b(nn: NN, X_0):
    iter_num = 10
    _, X = nn.forward_step(X_0, nn.weights[0], nn.biases[0])
    W = nn.weights[1]
    b = nn.biases[1]
    n, m = X.shape
    out_dimensions = nn.biases[1].shape[0]
    U = np.random.rand(out_dimensions, m)
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    d = np.random.rand(b.shape[0], b.shape[1])
    d /= np.linalg.norm(d)

    X_linear, X_forward = nn.forward_step(X, W, b)
    X_forward_T = X_forward.T
    g_x = np.dot(X_forward_T, U).item()


    Jb = np.diag(nn.activation_gradient(X_linear).flatten())
    g_grad = Jb.T @ U
    for i, epsilon in enumerate(epsilons):
        b_diff = b.copy()
        b_diff += d * epsilon
        _, X_eps_forward = nn.forward_step(X, W, b_diff)
        X_eps_forward_T = X_eps_forward.T
        gx_epsilon = np.dot(X_eps_forward_T, U).item()
        d_flat = d.reshape(-1, 1)
        diff[i] = abs(gx_epsilon - g_x)
        diff_grad[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ g_grad)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('W Jacobian Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()


if __name__ == '__main__':
    nn = NN([3, 5, 7, 8], tanh, tanh_grad)
    X = np.random.rand(3, 1)
    jacobian_test_layer_b(nn, X)
    # jacobian_test_layer_X(nn, X)
    # jacobian_test_layer_W(nn, X)
