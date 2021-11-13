from softmax import *
from NN import NN
import numpy as np
import matplotlib.pyplot as plt

epsilons = [np.power(0.5, i) for i in range(1, 10)]
rangeArr = [i for i in range(0, len(epsilons))]


def tan_h_gradient(x):
    return 1 - (np.tanh(x) ** 2)


def get_result_and_grads(test, X, hot_vec, nn: NN, l):
    if test == 'loss':
        nonlinear_res, probs = nn.softmax_layer(X, nn.weights[l], nn.biases[l], hot_vec)
        dW, db, _ = nn.softmax_gradient(probs, nn.weights[l], hot_vec, X)
    else:
        linear_res, nonlinear_res = nn.forward_step(X, nn.weights[l], nn.biases[l])
        n, batch_size = linear_res.shape
        # for single layer check we use 1 matrix
        dH_next = np.ones((n, batch_size))
        dW, db, _ = nn.hidden_layer_grad(X, nn.weights[l], nn.biases[l], dH_next)
    return nonlinear_res, dW, db


def jacobian_check(test='loss', batch_size=1, L=1):
    print("run tests")
    if test != 'all':
        L = 1
    full_test = test
    # chose layers sizes
    layers_dim = list(np.random.randint(2, 25, L + 1))
    # layers_dim = [2,5]
    input_size = layers_dim[0]
    number_of_labels = output_size = layers_dim[-1]
    # init wights
    nn_model = NN(layers_dim, np.tanh, softmax_regression, tan_h_gradient, softmax_grad)
    # init labels and samples
    # X = np.random.randn(input_size, batch_size)
    Y = np.random.choice(range(number_of_labels), size=batch_size)
    y_hot_vec = np.zeros((number_of_labels, batch_size))
    y_hot_vec[Y, np.arange(batch_size)] = 1
    # init perpetuated parameters
    deltaTheta = [np.random.random(theta.shape) for theta in nn_model.weights]
    deltaBias = [np.random.random(len(bias)) for bias in nn_model.biases]
    original_thetas = nn_model.weights.copy()
    original_bias = nn_model.biases.copy()
    if full_test == 'all':
        test = 'hidden'
    for l in range(L):
        last_res = last_res_p = 0
        firstOrderArr = []
        secondOrderArr = []
        if (full_test == 'all') and (l == L - 1):
            test = 'loss'
        X = np.random.randn(nn_model.weights[l].shape[1], batch_size)  # init last hidden layer
        result, dW, db = get_result_and_grads(test=test, X=X, hot_vec=y_hot_vec, nn=nn_model, l=l)
        for eps in epsilons:
            # replace the weight with perpetuated weight
            nn_model.weights[l] = original_thetas[l] + eps * deltaTheta[l]
            result_pert, _, _ = get_result_and_grads(test=test, X=X, hot_vec=y_hot_vec, nn=nn_model, l=l)
            if test == 'hidden':
                grad_delta = np.sum((eps * deltaTheta[l]) * dW, axis=1, keepdims=True)
            else:
                grad_delta = ((eps * deltaTheta[l]).reshape(-1, 1).T @ dW.reshape(-1, 1)).item()

            differance = np.linalg.norm(last_res / (result_pert - result))
            last_res = (result_pert - result)
            differance_prt = np.linalg.norm(last_res_p / (result_pert - result - grad_delta))
            last_res_p = (result_pert - result - grad_delta)
            firstOrderArr.append(differance)
            secondOrderArr.append(differance_prt)

        plt.plot(rangeArr, firstOrderArr, label="first-order")
        plt.plot(rangeArr, secondOrderArr, label="second-order")
        plt.yscale("log")
        plt.legend(loc='lower left', borderaxespad=0.)
        plt.xlabel('epsilons')
        plt.ylabel('differance')
        plt.title(f'Gradient Test For W{l}')
        plt.show()
        # same for bias
        nn_model.weights = original_thetas.copy()
        last_res = last_res_p = 0
        firstOrderArr = []
        secondOrderArr = []

        for eps in epsilons:
            # replace the weight with perpetuated weight
            nn_model.biases[l] = original_bias[l] + eps * deltaBias[l].reshape(len(nn_model.biases[l]), 1)
            result_pert, _, _ = get_result_and_grads(test=test, X=X, hot_vec=y_hot_vec, nn=nn_model, l=l)
            if test == 'hidden':
                grad_delta = np.sum((eps * deltaBias[l]) * db, axis=1, keepdims=True)
            else:
                grad_delta = ((eps * deltaBias[l]).reshape(-1, 1).T @ db.reshape(-1, 1)).item()
            differance = np.linalg.norm(last_res / (result_pert - result))
            last_res = (result_pert - result)
            differance_prt = np.linalg.norm(last_res_p / (result_pert - result - grad_delta))
            last_res_p = (result_pert - result - grad_delta)
            firstOrderArr.append(differance)
            secondOrderArr.append(differance_prt)
        plt.plot(rangeArr, firstOrderArr, label="first-order")
        plt.plot(rangeArr, secondOrderArr, label="second-order")
        plt.yscale("log")
        plt.legend(loc='lower left', borderaxespad=0.)
        plt.xlabel('epsilons')
        plt.ylabel('differance')
        plt.title(f'Gradient Test For b{l}')
        plt.show()
        nn_model.biasArray = original_bias.copy()


if __name__ == "__main__":
    np.random.seed(40)
    jacobian_check(test='hidden', batch_size=10)
    jacobian_check(test='loss', batch_size=1)
    jacobian_check(test='all', L=6, batch_size=1)
