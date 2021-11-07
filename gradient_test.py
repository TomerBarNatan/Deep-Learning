import numpy as np
import matplotlib.pyplot as plt
from softmax import softmax_regression, softmax_grad
from scipy.io import loadmat
num_of_iterations = 10
eps = [np.power(0.5, i) for i in range(num_of_iterations)]
number_of_butches = 100

def grad_W_test(X, W, bias, C):
    cost = softmax_regression(X, W, bias, C)
    grad_W, _ = softmax_grad(X, W, bias, C)
    grad_flat = grad_W.flatten()

    d = np.random.rand(grad_W.shape[0], grad_W.shape[1])
    d /= np.linalg.norm(d)
    d_flat = d.flatten()


    first_order = []
    second_order = []
    for epsilon in eps:
        curr_cost = softmax_regression(X, W + epsilon * d, bias, C)
        first_order.append(abs(curr_cost - cost))
        second_order.append(abs(curr_cost - cost -epsilon * d_flat.T @ grad_flat))
    return first_order, second_order


def draw_results(y_0, y_1):
    plt.semilogy([i for i in range(num_of_iterations)], y_0)
    plt.semilogy([i for i in range(num_of_iterations)], y_1)
    plt.legend(["First order approx", "Second order approx"])
    plt.title("Successful grad test in semilog scale plot")
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.show()


def extract_data():
    peaks_data = loadmat("PeaksData.mat")
    train_set = np.array(peaks_data['Yt'])
    train_set_labels = np.array(peaks_data['Ct'])

    idx = np.random.permutation(len(train_set_labels[0]))
    train_set_X = train_set[:, idx]
    train_set_C = train_set_labels[:, idx]

    train_set_X_batches = np.array_split(train_set_X, number_of_butches, axis=1)
    train_set_C_batches = np.array_split(train_set_C, number_of_butches, axis=1)

    output_size = len(train_set_C_batches[0][:, 0])
    input_size = train_set_X_batches[0].shape[0]
    bias = np.zeros([1, output_size])
    W = np.random.rand(input_size, output_size)
    return train_set_X_batches, W, bias, train_set_C_batches

if __name__ == '__main__':
    X_batches, W, bias, C_batches = extract_data()
    first_order, second_order = grad_W_test(X_batches[0], W, bias,  C_batches[0])
    draw_results(first_order, second_order)