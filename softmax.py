import numpy as np


def softmax_regression(X_L, W, bias, C):
    """
    The softmax regression activation function. Calculates the loss of the model.
    """
    probs = sigmoid(X_L, W, bias)
    m = X_L.shape[1]
    cost = (-1 / m) * (np.sum(C.T * np.log(probs)))
    return cost, probs


def sigmoid(X_L, W, bias):
    """
    The sigmoid function. Calculates a probability for each data sample to be each label.
    Returns a matrix where each column is the probabilities vector of a data point.
    """
    output_layer = X_L.T @ W + bias
    output_layer -= np.max(output_layer)
    probs = (np.exp(output_layer).T / np.sum(np.exp(output_layer), axis=1)).T
    return probs


def softmax_grad(X, W, bias, C):
    """
    Gradient of the softmax regression according to the weights, the bias and the data.
    """
    probs = sigmoid(X, W, bias)
    m = X.shape[1]
    grad_W = (-1 / m) * (X @ (C.T - probs))
    grad_b = (-1 / m) * (np.sum(C.T - probs, axis=0)).T
    grad_X = (1 / m) * W @ (probs.T - C)
    return grad_W, grad_b, grad_X
