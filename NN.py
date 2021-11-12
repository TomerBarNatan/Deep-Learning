import numpy as np


class NN:
    def __init__(self, network_layers_list, activation, cost_function, activation_gradient, last_gradient):
        self.weights = []
        self.biases = []
        self.activation = activation
        self.cost_function = cost_function
        self.activation_gradient = activation_gradient
        self.last_gradient = last_gradient

        for i in range(len(network_layers_list) - 1):
            W_i = np.random.randn(network_layers_list[i + 1], network_layers_list[i])
            bias_i = np.zeros([network_layers_list[i + 1], 1])
            self.weights.append(W_i)
            self.biases.append(bias_i)

    def forward_step(self, x, W, bias):
        output = self.activation(W @ x + bias)
        return output

    def forward(self, x_0, C):
        current_x = x_0
        for i in range(len(self.weights) - 1):
            current_x = self.forward_step(current_x, self.weights[i], self.biases[i])
        cost, probs = self.cost_function(current_x, self.weights[-1], self.biases[-1], C)
        return cost, probs

    def hidden_layer_grad(self, X, W, b, v):
        grad_activation = self.activation_gradient(W @ X + b)
        grad_b = grad_activation * v
        grad_W = grad_b @ X.T
        new_v = W.T @ grad_b
        return grad_W, grad_b, new_v

    def backpropagation(self, X_list, C):
        layer_number = len(X_list)
        weight_grads = []
        bias_grads = []

        # last layer gradient
        W_grad, b_grad, x_grad = self.last_gradient(X_list[-1], self.weights[-1], self.biases[-1], C)
        weight_grads.append(W_grad.copy())
        bias_grads.append(b_grad.copy())
        weight_grads.append(W_grad)
        bias_grads.append(b_grad)
        v_i = x_grad.copy()

        # hidden layer grads
        for i in range(layer_number - 1, 0, -1):
            F_grad_W_i, F_grad_b_i, v_i = self.hidden_layer_grad(X_list[i], self.weights[i], self.biases[i], v_i)
            weight_grads.append(F_grad_W_i.copy())
            bias_grads.append(F_grad_b_i.copy())
        return weight_grads, bias_grads

    def update_thetas(self, W_grad_list, bias_grad_list, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learning_rate * W_grad_list[i]
            self.biases[i] = self.biases[i] - learning_rate * bias_grad_list[i]
