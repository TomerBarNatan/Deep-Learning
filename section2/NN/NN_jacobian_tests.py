import numpy as np
from nn_from_scratch.section2.NN.NN import NN
from nn_from_scratch.section1.softmax import softmax_regression
from nn_from_scratch.section2.activations import tanh, tanh_grad
from matplotlib.pyplot import plot as plt

epsilons = [np.power(0.5, i) for i in range(1, 10)]


def jacobian_test(func, batch_size, layers):
    layers_dim = list(np.random.randint(2, 25, layers + 1))
    input_size = layers_dim[0]
    number_of_labels = output_size = layers_dim[-1]

    model = NN(layers_dim, tanh, tanh_grad)

    Y = np.random.choice(range(number_of_labels), size=batch_size)
    y_hot_vec = np.zeros((number_of_labels, batch_size))
    y_hot_vec[Y, np.arange(batch_size)] = 1

    deltaTheta = [np.random.random(theta.shape) for theta in model.weights]
    deltaBias = [np.random.random(len(bias)) for bias in model.biases]
    original_thetas = model.weights.copy()
    original_bias = model.biases.copy()


def jacobian_test_layer_X(nn: NN, X):
    n, m = X.shape
    out_dimensions = nn.biases[0].shape[0]
    U = np.random.rand(out_dimensions, m)
    v = np.random.rand(out_dimensions, m)  # figure out dimensions
    iter_num = 10
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    d = np.random.rand(*X.shape)
    d /= np.linalg.norm(d)
    _, X_forward_T = nn.forward_step(X, nn.weights[0], nn.biases[0]).T
    g_x = np.dot(X_forward_T, U).item()
    nn.backward_hidden_layer([X, X_forward_T.T], 1, v)
    JacTu_X = nn.hidden_layer_grad(X, nn.weights[0], nn.biases[0], v)
    for i, epsilon in enumerate(epsilons):
        X_diff = X.copy()
        X_diff += d * epsilon
        _, X_eps_forward_T = nn.forward_step(X_diff, nn.weights[0], nn.biases[0]).T
        gx_epsilon = np.dot(X_eps_forward_T, U).item()
        d_flat = d.reshape(-1, 1)
        JacTu_X_flat = JacTu_X.reshape(-1, 1)
        diff[i] = abs(gx_epsilon - g_x)
        diff_grad[i] = abs(gx_epsilon - g_x - epsilon * d_flat.T @ JacTu_X_flat)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('X Jacobian Test Results')
    plt.legend(("diff without grad", "diff with grad"))

    # for l in range(layers):
    #     last_res = last_res_p = 0
    #     firstOrderArr = []
    #     secondOrderArr = []
    #     X = np.random.randn(model.weights[l].shape[1], batch_size)
    #     cost, probs = softmax_regression(X, deltaTheta[l], deltaBias[l], y_hot_vec)
