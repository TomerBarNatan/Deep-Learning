import numpy as np
import matplotlib.pyplot as plt
from softmax import softmax_regression, softmax_grad
from loadData import extract_data

num_of_iterations = 10


def grad_W_test(X, W, bias, C):
    # calculate F(W) and the gradient w.r.t W
    F_0 = softmax_regression(X, W, bias, C)
    grad_W, _ = softmax_grad(X, W, bias, C)
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
        F_k = softmax_regression(X, W + epsilon * d, bias, C)
        F_1 = F_0 + epsilon * d_flat.T @ grad_W_flat
        zero_order.append(abs(F_k - F_0))
        first_order.append(abs(F_k - F_1))
        print(k, '\t', abs(F_k - F_0), '\t', abs(F_k - F_1))
    return zero_order, first_order


def grad_bias_test(X, W, bias, C):
    # calculate F(W) and the gradient w.r.t bias
    F_0 = softmax_regression(X, W, bias, C)
    _, grad_bias = softmax_grad(X, W, bias, C)

    # define a random d for the test
    d = np.random.rand(grad_bias.shape[0])
    d /= np.linalg.norm(d)

    zero_order = []
    first_order = []
    print('\nGradient test w.r.t bias results:')
    print('k\t\terror order 0\t\terror order 1')
    for k in range(num_of_iterations):
        epsilon = 0.5 ** k
        F_k = softmax_regression(X, W, bias + epsilon * d, C)
        F_1 = F_0 + epsilon * d.T @ grad_bias
        zero_order.append(abs(F_k - F_0))
        first_order.append(abs(F_k - F_1))
        print(k, '\t', abs(F_k - F_0), '\t', abs(F_k - F_1))
    return zero_order, first_order


def draw_results(y_0, y_1, result_for='weights'):
    plt.semilogy([i for i in range(num_of_iterations)], y_0)
    plt.semilogy([i for i in range(num_of_iterations)], y_1)
    plt.legend(["Zero order approx", "First order approx"])
    plt.title("Successful {res_for} gradient test in semilog scale".format(res_for=result_for))
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.show()


if __name__ == '__main__':
    X_batches, W, bias, C_batches = extract_data("PeaksData.mat", 100)
    zero_order_W, first_order_W = grad_W_test(X_batches[0], W, bias, C_batches[0])
    draw_results(zero_order_W, first_order_W)
    zero_order_bias, first_order_bias = grad_bias_test(X_batches[0], W, bias, C_batches[0])
    draw_results(zero_order_bias, first_order_bias, result_for='bias')