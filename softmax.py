import numpy as np


def softmax_regression(X_L, W, bias, c):
    prob = sigmoid(X_L, W)
    m = X_L.shape[1]
    cost = (-1 / m) * (np.sum(c.T * np.log(prob), axis = 0))

    return cost


def sigmoid(X_L, W):
    output_layer = X_L.T @ W
    output_layer -= np.max(output_layer)
    probability = np.exp(output_layer) / np.sum(np.exp(output_layer), axis=1)
    return probability


def softmax_gradient(X_L, W, c):
    m = X_L.shape[1]
    prob = sigmoid(X_L, W)
    gradient = (-1/m)*X_L@(prob - c.T)