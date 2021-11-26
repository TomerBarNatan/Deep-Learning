import numpy as np
import matplotlib.pyplot as plt
from softmax import softmax_regression, softmax_grad
from nn_from_scratch.loadData import extract_grad_test_data

num_of_iterations = 10


def grad_W_test(X, W, bias, C):
    # calculate F(W) and the gradient w.r.t W
    F_0, _ = softmax_regression(X, W, bias, C)
    grad_W, _, _ = softmax_grad(X, W, bias, C)
    grad_W_flat = grad_W.flatten()

    # define a random d for the test
    d = np.random.rand(grad_W.shape[0], grad_W.shape[1])
    d /= np.linalg.norm(d)
    d_flat = d.flatten()

    zero_order = []
    first_order = []
    print('\nGradient test w.r.t W results:')
    print('k\t\terror order 0\t\terror order 1')
    for k in range(num_of_iterations):
        epsilon = 0.5 ** k
        F_k, _ = softmax_regression(X, W + epsilon * d, bias, C)
        F_1 = F_0 + epsilon * d_flat.T @ grad_W_flat
        zero_order.append(abs(F_k - F_0))
        first_order.append(abs(F_k - F_1))
        print(k, '\t', abs(F_k - F_0), '\t', abs(F_k - F_1))
    return zero_order, first_order


def grad_bias_test(X, W, bias, C):
    # calculate F(W) and the gradient w.r.t bias
    F_0, _ = softmax_regression(X, W, bias, C)
    _, grad_bias, _ = softmax_grad(X, W, bias, C)

    # define a random d for the test
    d = np.random.rand(grad_bias.shape[0])
    d /= np.linalg.norm(d)

    zero_order = []
    first_order = []
    print('\nGradient test w.r.t bias results:')
    print('k\t\terror order 0\t\terror order 1')
    for k in range(num_of_iterations):
        epsilon = 0.5 ** k
        F_k, _ = softmax_regression(X, W, bias + epsilon * d, C)
        F_1 = F_0 + epsilon * d.T @ grad_bias
        zero_order.append(abs(F_k - F_0))
        first_order.append(abs(F_k - F_1))
        print(k, '\t', abs(F_k - F_0), '\t', abs(F_k - F_1))
    return zero_order, first_order


def hidden_layer_grad_test(nn, X):
    C = np.zeros((10, 1))
    C[0] = 1
    # calculate F(W) and the gradient w.r.t bias
    cost, probs, linear_layers, nonlinear_layers = nn.forward(X, C)
    _, _, x_grad = nn.softmax_gradient(nonlinear_layers[-1],nn.weights[-1],C,nonlinear_layers[-2])
    grad_W, _, _ = nn.hidden_layer_grad(nonlinear_layers[1], nn.weights[1], nn.biases[1], x_grad)
    F_0 = nonlinear_layers[2]
    # define a random d for the test
    d = np.random.rand(grad_W.shape[0],1)
    # d /= np.linalg.norm(d)
    d_flat = d.flatten()
    grad_W_flat = grad_W.flatten()

    zero_order = []
    first_order = []
    print('\nGradient test w.r.t bias results:')
    print('k\t\terror order 0\t\terror order 1')
    for k in range(num_of_iterations):
        epsilon = 0.5 ** k
        _, F_k = nn.forward_step(nonlinear_layers[1], nn.weights[1] + epsilon * d, nn.biases[1])
        F_1 = F_0 + epsilon * d.T @ grad_W
        zero_order.append(np.linalg.norm(F_k - F_0))
        first_order.append(np.linalg.norm((F_k - F_1)))
        print(k, '\t', np.linalg.norm(F_k - F_0), '\t', np.linalg.norm(F_k - F_1))
    return zero_order, first_order


def draw_results(y_0, y_1, result_for='Weights'):
    plt.semilogy([i for i in range(num_of_iterations)], y_0)
    plt.semilogy([i for i in range(num_of_iterations)], y_1)
    plt.legend(["Zero order approx", "First order approx"])
    plt.title("Successful {res_for} Gradient Test In Semilog Scale".format(res_for=result_for))
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()


epsilonArray = [np.power(0.5, i) for i in range(0, 10)]
rangeArr = [i for i in range(0, 10)]


# def hidden_layer_grad_test2(nn, X):
#     C = np.zeros((10, 1))
#     C[0] = 1
#     firstOrderArr = []
#     secondOrderArr = []
#     cost, probs, linear_layers, nonlinear_layers = nn.forward(X, C)
#     _, _, x_grad = nn.softmax_gradient(nonlinear_layers[-1],nn.weights[-1],C,nonlinear_layers[-2])
#     grad_W, grad_b, _ = nn.hidden_layer_grad(nonlinear_layers[1], nn.weights[1], nn.biases[1], x_grad)
#     F_0 = nonlinear_layers[2]
#     d = np.random.rand(grad_b.shape[0], grad_b.shape[1])
#     d = d / np.linalg.norm(d)
#     d_vec = d.flatten()
#     grad = grad_b.flatten()
#     for eps in epsilonArray:
#         d_theta = nn.biases[1].copy()
#         d_theta = d_theta + d * eps
#         linear, d_F_0 = nn.forward_step(nonlinear_layers[1], nn.weights[1], d_theta)
#         firstOrderArr.append(np.linalg.norm(d_F_0 - F_0))
#         secondOrderArr.append(np.linalg.norm(d_F_0 - F_0 - eps * d_vec.T @ grad))
#     plt.plot(rangeArr, firstOrderArr, label="first-order")
#     plt.plot(rangeArr, secondOrderArr, label="second-order")
#     plt.yscale("log")
#     plt.xscale("log")
#     plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',
#     ncol=2, mode="expand", borderaxespad=0.)
#     plt.xlabel('epsilons')
#     plt.ylabel('absolute differance')
#     plt.title('wights gradient test:')
#     plt.show()


if __name__ == '__main__':
    X_batches, W, bias, C_batches = extract_grad_test_data("PeaksData.mat", 100)
    zero_order_W, first_order_W = grad_W_test(X_batches[0], W, bias, C_batches[0])
    draw_results(zero_order_W, first_order_W)
    zero_order_bias, first_order_bias = grad_bias_test(X_batches[0], W, bias, C_batches[0])
    draw_results(zero_order_bias, first_order_bias, result_for='Biases')
    # nn = NN([50, 100, 20,10], tanh, tanh_grad)
    # X = np.random.rand(50, 1)
    # zero_order_bias, first_order_bias = hidden_layer_grad_test(nn, X)
    # # hidden_layer_grad_test2(nn, X)
    # draw_results(zero_order_bias, first_order_bias, result_for='Weights')
