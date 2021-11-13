import numpy as np
from NN import NN
from softmax import softmax_regression, softmax_grad

epsilons = [np.power(0.5, i) for i in range(1, 10)]

def tan_h_gradient(x):
    return 1 - np.tanh(x) ** 2


def jacobian_test(func, batch_size, layers):
    layers_dim = list(np.random.randint(2, 25, layers + 1))
    input_size = layers_dim[0]
    number_of_labels = output_size = layers_dim[-1]

    model = NN(layers_dim, np.tanh, softmax_regression, tan_h_gradient, softmax_grad)

    Y = np.random.choice(range(number_of_labels), size=batch_size)
    y_hot_vec = np.zeros((number_of_labels, batch_size))
    y_hot_vec[Y, np.arange(batch_size)] = 1

    deltaTheta = [np.random.random(theta.shape) for theta in model.weights]
    deltaBias = [np.random.random(len(bias)) for bias in model.biases]
    original_thetas = model.weights.copy()
    original_bias = model.biases.copy()

    for l in range(layers):
        last_res = last_res_p = 0
        firstOrderArr = []
        secondOrderArr = []
        X = np.random.randn(model.weights[l].shape[1], batch_size)
        cost, probs = softmax_regression(X, deltaTheta[l], deltaBias[l], y_hot_vec)
