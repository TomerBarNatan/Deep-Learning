import numpy as np
import math


class NN:
    """
    NN is a class which represents the neural network.
    """
    def __init__(self, network_layers_list, activation, activation_gradient):
        self.weights = []
        self.biases = []
        self.activation = activation
        self.activation_gradient = activation_gradient

        np.random.seed(0)
        scale = 1 / max(1., (2 + 2) / 2.)
        limit = math.sqrt(3.0 * scale)

        for i in range(len(network_layers_list) - 1):
            W_i = np.random.uniform(-limit, limit, size=(network_layers_list[i + 1], network_layers_list[i]))
            W_i /= np.linalg.norm(W_i)
            # W_i = np.zeros((network_layers_list[i + 1], network_layers_list[i]))
            # W_i[1:2] = 1
            bias_i = np.zeros([network_layers_list[i + 1], 1])
            self.weights.append(W_i)
            self.biases.append(bias_i)

    def softmax_layer(self, X_L, W, bias, C):
        """
        The softmax function fot the last layer in the network
        :param X_L: the data from step n-1
        :param W: the weights from step n-1
        :param bias: the bias from step n-1
        :param C: the indicators
        :return: cost value (scalar) and the probabilities matrix (for each data point, a vector of probabilities to get
        each label.
        """
        batch_size = C.shape[1]
        scores = W @ X_L + bias
        scores -= np.max(scores)
        probs = (np.exp(scores) / np.sum(np.exp(scores), axis=0))
        cost = (-1 / batch_size) * (np.sum(np.log(probs) * C))
        return cost, probs

    def forward_step(self, x, W, bias):
        """
        A forward step in the network, alters the data in transition from layer k to layer k+1
        :param x: the data in layer k
        :param W: the weights in layer k
        :param bias: the bias in layer k
        :return: the linear result of W@x + bias, and f(linear) where f is the configured activation function
        """
        linear = W @ x + bias
        nonlinear = self.activation(linear)
        return linear, nonlinear

    def forward(self, x_0, C):
        """
        Forward routine of the whole network, starting from layer 1 and performing n forward steps (including the last
        step of the softmax function)
        :param x_0: the initial data
        :param C: the indicators
        :return: cost of the entire network, probabilities matrix, linear (W@x+bias) and non-linear (activation(linear))
        results for each layer (as lists).
        """
        linear_layers = [x_0.copy()]
        nonlinear_layers = [x_0.copy()]
        current_x = x_0
        for i in range(len(self.weights) - 1):
            linear, current_x = self.forward_step(current_x, self.weights[i], self.biases[i])
            linear_layers.append(linear.copy())
            nonlinear_layers.append(current_x.copy())

        cost, probs = self.softmax_layer(current_x, self.weights[-1], self.biases[-1], C)
        nonlinear_layers.append(probs.copy())
        return cost, probs, linear_layers, nonlinear_layers

    def softmax_gradient(self, X, W, C, v):
        """
        Calculate the softmax gradients w.r.t the weights, the bias and the data.
        :param X: the data
        :param W: the weights
        :param C: the indicators
        :param v: a vector v
        :return: softmax gradient w.r.t weights, bias and data
        """
        batch_size = C.shape[1]
        dl_dy = (1 / batch_size) * (X - C)
        dl_W = dl_dy @ v.T
        dl_db = np.sum(dl_dy, axis=1, keepdims=True)
        new_v = W.T @ dl_dy
        return dl_W, dl_db, new_v

    def hidden_layer_grad(self, X, W, b, v):
        """
        Calculate the gradient of an individual hidden layer.
        :param X: the current layer's data
        :param W: the current layer's weights
        :param b: the current layer's bias
        :param v: a vector v (from the next layer)
        :return: The hidden layer gradient w.r.t weights, bias and the new v vector to use in the previous layer
        """
        linear = W @ X + b
        batch_size = linear.shape[1]
        grad_activation = self.activation_gradient(linear)
        common = grad_activation * v
        grad_W = (1 / batch_size) * common @ X.T
        grad_b = (1 / batch_size) * np.sum(common, axis=1, keepdims=True)
        new_v = W.T @ common
        return grad_W, grad_b, new_v

    def backpropagation(self, X_list, C):
        """
        The backpropagation process of the network.
        :param X_list: a list of the data X in each layer
        :param C: the indicators
        :return: gradients list of each layer w.r.t weights and bias
        """
        layer_number = len(X_list)
        weight_grads = []
        bias_grads = []

        # last layer gradient
        W_grad, b_grad, x_grad = self.softmax_gradient(X_list[-1], self.weights[-1], C, X_list[-2])
        weight_grads.append(W_grad.copy())
        bias_grads.append(b_grad.copy())
        v_i = x_grad.copy()

        # hidden layer grads
        for i in range(layer_number - 2, 0, -1):
            F_grad_W_i, F_grad_b_i, v_i = self.hidden_layer_grad(X_list[i - 1], self.weights[i - 1], self.biases[i - 1],
                                                                 v_i)
            weight_grads.append(F_grad_W_i.copy())
            bias_grads.append(F_grad_b_i.copy())
        return list(reversed(weight_grads)), list(reversed(bias_grads))

    def update_thetas(self, W_grad_list, bias_grad_list, learning_rate):
        """
        Update the weights and biases of the network
        :param W_grad_list: list of gradients w.r.t weights
        :param bias_grad_list: list of gradients w.r.t bias
        :param learning_rate: the learning rate
        """
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learning_rate * W_grad_list[i]
            self.biases[i] = self.biases[i] - learning_rate * bias_grad_list[i]
