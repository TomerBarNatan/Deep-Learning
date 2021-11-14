import numpy as np
import math


class NN:
    def __init__(self, network_layers_list, activation, cost_function, activation_gradient, last_gradient):
        self.weights = []
        self.biases = []
        self.activation = activation
        self.cost_function = cost_function
        self.activation_gradient = activation_gradient
        self.last_gradient = last_gradient

        np.random.seed(0)
        scale = 1 / max(1., (2 + 2) / 2.)
        limit = math.sqrt(3.0 * scale)

        for i in range(len(network_layers_list) - 1):
            W_i = np.random.uniform(-limit, limit, size=(network_layers_list[i + 1], network_layers_list[i]))
            bias_i = np.zeros([network_layers_list[i + 1], 1])
            self.weights.append(W_i)
            self.biases.append(bias_i)

    def softmax_layer(self, X_L, W, bias, C):
        batch_size = C.shape[1]

        scores = W @ X_L + bias
        scores -= np.max(scores)
        probs = (np.exp(scores) / np.sum(np.exp(scores), axis=0))
        cost = (-1 / batch_size) * (np.sum(np.log(probs) * C))
        return cost, probs

    def forward_step(self, x, W, bias):
        linear = W @ x + bias
        nonlinear = self.activation(linear)
        return linear, nonlinear

    def forward(self, x_0, C):
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
        batch_size = C.shape[1]

        dl_dy = (1 / batch_size) * (X - C)
        dl_W = dl_dy @ v.T
        dl_db = np.sum(dl_dy, axis=1, keepdims=True)
        new_v = W.T @ dl_dy
        return dl_W, dl_db, new_v

    def hidden_layer_grad(self, X, W, b, v):
        linear = W @ X + b
        batch_size = linear.shape[1]
        grad_activation = self.activation_gradient(linear)
        common = grad_activation * v
        grad_W = (1 / batch_size) * common @ X.T
        grad_b = (1 / batch_size) * np.sum(common, axis=1, keepdims=True)
        new_v = W.T @ common
        return grad_W, grad_b, new_v

    def backpropagation(self, X_list, C):
        layer_number = len(X_list)
        weight_grads = []
        bias_grads = []

        # last layer gradient
        W_grad, b_grad, x_grad = self.softmax_gradient(X_list[-1], self.weights[-1], C, X_list[-2])
        weight_grads.append(W_grad.copy())
        bias_grads.append(b_grad.copy())
        weight_grads.append(W_grad)
        bias_grads.append(b_grad)
        v_i = x_grad.copy()

        # hidden layer grads
        for i in range(layer_number - 2, 0, -1):
            F_grad_W_i, F_grad_b_i, v_i = self.hidden_layer_grad(X_list[i-1], self.weights[i-1], self.biases[i-1], v_i)
            weight_grads.append(F_grad_W_i.copy())
            bias_grads.append(F_grad_b_i.copy())
        return weight_grads, bias_grads

    def update_thetas(self, W_grad_list, bias_grad_list, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learning_rate * W_grad_list[i]
            self.biases[i] = self.biases[i] - learning_rate * bias_grad_list[i]
