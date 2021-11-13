import numpy as np
import matplotlib.pyplot as plt
from softmax import softmax_regression, softmax_grad
from loadData import extract_grad_test_data
from NN import NN

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


# def hidden_layer_grad_test(nn, X):
#     # calculate F(W) and the gradient w.r.t bias
#     linear, F_0 = nn.forward_step(X, nn.weights[0], nn.biases[0])
#     v = np.ones(linear.shape)
#     grad_W, _, _ = nn.hidden_layer_grad(X, nn.weights[0], nn.biases[0], v)
#
#     # define a random d for the test
#     d = np.random.rand(grad_W.shape[0], grad_W.shape[1])
#     d /= np.linalg.norm(d)
#     d_flat = d.flatten()
#     grad_W_flat = grad_W.flatten()
#
#
#     zero_order = []
#     first_order = []
#     print('\nGradient test w.r.t bias results:')
#     print('k\t\terror order 0\t\terror order 1')
#     for k in range(num_of_iterations):
#         epsilon = 0.5 ** k
#         _, F_k = nn.forward_step(X, nn.weights[0] + epsilon * d, nn.biases[0])
#         F_1 = F_0 + epsilon * d_flat.T @ grad_W_flat
#         zero_order.append(np.linalg.norm(F_k - F_0))
#         first_order.append(np.linalg.norm(F_k - F_1))
#         print(k, '\t', np.linalg.norm(F_k - F_0), '\t', np.linalg.norm(F_k - F_1))
#     return zero_order, first_order


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

def hidden_layer_grad_test(nn, X):
     firstOrderArr = []
     secondOrderArr = []
     linear, F_0 = nn.forward_step(X, nn.weights[0], nn.biases[0])
     v = np.random.rand(linear.shape[0],linear.shape[1])
     grad_W, _, _ = nn.hidden_layer_grad(X,nn.weights[0], nn.biases[0], v)
     d = np.random.rand(grad_W.shape[0], grad_W.shape[1])
     d = d / np.linalg.norm(d)
     d_vec = d.flatten()
     grad = grad_W.flatten()
     for eps in epsilonArray:
         d_theta = nn.weights[0].copy()
         d_theta = d_theta + d * eps
         linear, d_F_0 = nn.forward_step(X, d_theta, nn.biases[0])
         firstOrderArr.append(np.linalg.norm(d_F_0 - F_0))
         secondOrderArr.append(np.linalg.norm(d_F_0 - F_0 - eps * d_vec.T @ grad))
     plt.plot(rangeArr, firstOrderArr, label="first-order")
     plt.plot(rangeArr, secondOrderArr, label="second-order")
     plt.yscale("log")
     plt.xscale("log")
     plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',
     ncol=2, mode="expand", borderaxespad=0.)
     plt.xlabel('epsilons')
     plt.ylabel('absolute differance')
     plt.title('wights gradient test:')
     plt.show()

def tan_h_gradient(x):
    return 1 - (np.tanh(x) ** 2)


if __name__ == '__main__':
    # X_batches, W, bias, C_batches = extract_grad_test_data("PeaksData.mat", 100)
    # zero_order_W, first_order_W = grad_W_test(X_batches[0], W, bias, C_batches[0])
    # draw_results(zero_order_W, first_order_W)
    # zero_order_bias, first_order_bias = grad_bias_test(X_batches[0], W, bias, C_batches[0])
    # draw_results(zero_order_bias, first_order_bias, result_for='Biases')
    nn = NN([100, 1000, 100, 10], np.tanh, None, tan_h_gradient, None)
    X = np.random.rand(100, 1)
    hidden_layer_grad_test(nn, X)
    # draw_results(zero_order_bias, first_order_bias, result_for='Weights')
