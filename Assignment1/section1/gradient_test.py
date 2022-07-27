import numpy as np
import matplotlib.pyplot as plt
from softmax import softmax_regression, softmax_grad
from Assignment1.data.loadData import extract_grad_test_data

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
    draw_results(zero_order, first_order)


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
    draw_results(zero_order, first_order, 'Biases')


def draw_results(y_0, y_1, result_for='Weights'):
    plt.semilogy([i for i in range(num_of_iterations)], y_0)
    plt.semilogy([i for i in range(num_of_iterations)], y_1)
    plt.legend(["Zero order approx", "First order approx"])
    plt.title("Successful {res_for} Gradient Test In Semilog Scale".format(res_for=result_for))
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()


if __name__ == '__main__':
    X_batches, W, bias, C_batches = extract_grad_test_data("GMMData", 100)
    grad_W_test(X_batches[0], W, bias, C_batches[0])
    grad_bias_test(X_batches[0], W, bias, C_batches[0])
