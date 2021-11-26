import numpy as np
from NN import NN
from activations import tanh, tanh_grad
import matplotlib.pyplot as plt
num_of_iterations = 30


def nn_grad_test(nn: NN, X):
    C = np.zeros((10, 1))
    C[0] = 1
    F_0, _, _, nonlinear_layers = nn.forward(X, C)
    weight_grads, biases_grads = nn.backpropagation(nonlinear_layers, C)
    # define a random d for the test
    ds_w = [np.random.rand(w.shape[0],w.shape[1]) for w in nn.weights]
    ds_b = [np.random.rand(b.shape[0],b.shape[1]) for b in nn.biases]
    weight_grads_flatten = np.concatenate([w.flatten() for w in weight_grads])
    biases_grads_flatten = np.concatenate([b.flatten() for b in biases_grads])
    grads_flatten = np.concatenate([weight_grads_flatten,biases_grads_flatten])
    d_flat = np.random.rand(grads_flatten.shape[0], 1)
    d_flat /= np.linalg.norm(d_flat)
    nn_weights_original = nn.weights.copy()
    nn_biases_original = nn.biases.copy()

    zero_order = []
    first_order = []
    print('\nGradient test w.r.t bias results:')
    print('k\t\terror order 0\t\terror order 1')
    for k in range(num_of_iterations):
        epsilon = 0.5 ** k
        nn.weights = [nn.weights[i] + epsilon* ds_w[i] for i in range(len(nn.weights))]
        nn.biases = [nn.biases[i] + epsilon* ds_b[i] for i in range(len(nn.biases))]
        F_k, _, _, _ = nn.forward(X,C)
        F_1 = F_0 + epsilon * d_flat.T @ grads_flatten
        zero_order.append(np.linalg.norm(F_k - F_0))
        first_order.append(np.linalg.norm((F_k - F_1) ** 2))
        print(k, '\t', np.linalg.norm(F_k - F_0), '\t', np.linalg.norm(F_k - F_1))
        nn.weights = nn_weights_original
        nn.biases = nn_biases_original
    return zero_order, first_order


def draw_results(y_0, y_1, result_for='Weights'):
    plt.semilogy([i for i in range(num_of_iterations)], y_0)
    plt.semilogy([i for i in range(num_of_iterations)], y_1)
    plt.legend(["Zero order approx", "First order approx"])
    plt.title("Successful {res_for} Gradient Test In Semilog Scale".format(res_for=result_for))
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()


if __name__ == '__main__':
    nn = NN([50, 10], tanh, tanh_grad)
    X = np.random.rand(50, 1)
    zero_order_bias, first_order_bias = nn_grad_test(nn, X)
    draw_results(zero_order_bias, first_order_bias, result_for='Weights')