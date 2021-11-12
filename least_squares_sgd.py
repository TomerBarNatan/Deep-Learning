import matplotlib.pyplot as plt
from softmax import *


def sgd(grad_function, cost_function, X, W, C, bias, batch_size, learning_rate, iter_num):
    costs = []
    for iter in range(iter_num):
        shuffler = np.random.permutation(X.shape[1])
        X = X[shuffler]
        C = C[shuffler]
        for i in range(batch_size):
            X_i = X[i * batch_size:i * batch_size + batch_size, :]
            C_i = C[i * batch_size:i * batch_size + batch_size]
            grad = grad_function(X_i, W, C_i, bias)
            W = W - (1 / batch_size) * learning_rate * (grad)
        costs.append(cost_function(X, W, C, bias))
    return W, costs


def least_squares_gradient(A, x, b, bias=None):
    grad = 2 * A.T @ A @ x - 2 * A.T @ b
    return grad

def least_squares_cost_function(A, x, b, bias=None):
    cost = np.linalg.norm(A @ x - b)
    return cost

if __name__ == '__main__':
    iter_num = 1000
    A = np.random.rand(10, 6)
    b = np.random.rand(10, 1)
    x = np.random.rand(6, 1)
    x, costs = sgd(least_squares_gradient, least_squares_cost_function, A, x, b, None, 10, 1, iter_num)
    actual = np.linalg.lstsq(A, b)
    errors = [np.linalg.norm(W - actual[0]) for W in costs]
    plt.semilogy([i for i in range(iter_num)], costs)
    plt.title("Testing SGD with Least Squares problem")
    plt.xlabel("Iteration Number")
    plt.ylabel("Cost in logarithmic scale")
    plt.show()
