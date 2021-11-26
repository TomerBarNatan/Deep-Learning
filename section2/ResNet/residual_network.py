import numpy as np
import math


class ResNet:
    def __init__(self, input_size, num_of_layers, num_of_labels, activation, activation_gradient, first_layer=8):
        self.weights = []
        self.biases = []
        self.activation = activation
        self.activation_gradient = activation_gradient

        np.random.seed(0)
        scale = 1 / max(1., (2 + 2) / 2.)
        limit = math.sqrt(3.0 * scale)
        W_1 = np.random.uniform(-limit, limit, size=(first_layer, input_size))
        W_1 /= np.linalg.norm(W_1)
        bias_1 = np.zeros([first_layer, 1])
        self.weights.append(W_1)
        self.biases.append(bias_1)

        for i in range(1, num_of_layers - 2):
            W1_i = np.random.uniform(-limit, limit, size=(first_layer, first_layer))
            W1_i /= np.linalg.norm(W1_i)
            bias_i = np.zeros([first_layer, 1])
            W2_i = np.random.uniform(-limit, limit, size=(first_layer, first_layer))
            W2_i /= np.linalg.norm(W2_i)
            self.weights.append([W1_i, W2_i])
            self.biases.append(bias_i)

        W_n = np.random.uniform(-limit, limit, size=(num_of_labels, first_layer))
        bias_n = np.zeros([num_of_labels, 1])
        self.weights.append(W_n)
        self.biases.append(bias_n)

    def softmax_layer(self, X_L, W, bias, C):
        batch_size = C.shape[1]

        scores = W @ X_L + bias
        scores -= np.max(scores)
        probs = (np.exp(scores) / np.sum(np.exp(scores), axis=0))
        cost = (-1 / batch_size) * (np.sum(np.log(probs) * C))
        return cost, probs

    def res_forward_step(self, x, Ws, bias):
        linear = Ws[0] @ x + bias
        nonlinear = x + Ws[1] @ self.activation(linear)
        return linear, nonlinear

    def forward_step(self, x, W, bias):
        linear = W @ x + bias
        nonlinear = self.activation(linear)
        return linear, nonlinear

    def forward(self, x_0, C):
        linear_layers = [x_0.copy()]
        nonlinear_layers = [x_0.copy()]
        linear, current_x = self.forward_step(x_0, self.weights[0], self.biases[0])
        linear_layers.append(linear.copy())
        nonlinear_layers.append(current_x.copy())
        for i in range(1, len(self.weights) - 1):
            linear, current_x = self.res_forward_step(current_x, self.weights[i], self.biases[i])
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

    def res_hidden_layer_grad(self, X, Ws, b, v):
        linear = Ws[0] @ X + b
        batch_size = linear.shape[1]
        grad_activation = self.activation_gradient(linear)
        grad_W1 = (1 / batch_size) * (grad_activation * (Ws[1].T @ v)) @ X.T
        grad_W2 = (1 / batch_size) * v @ (self.activation(linear)).T
        grad_b = (1 / batch_size) * np.sum((grad_activation * (Ws[1].T @ v)), axis=1, keepdims=True)
        new_v = v + (Ws[0].T @ (grad_activation * (Ws[1].T @ v)))
        return grad_W1, grad_W2, grad_b, new_v

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
        v_i = x_grad.copy()

        # hidden layer grads
        for i in range(layer_number - 2, 1, -1):
            F_grad_W1_i, F_grad_W2_i, F_grad_b_i, v_i = self.res_hidden_layer_grad(X_list[i - 1], self.weights[i - 1],
                                                                                   self.biases[i - 1],
                                                                                   v_i)
            weight_grads.append([F_grad_W1_i.copy(), F_grad_W2_i.copy()])
            bias_grads.append(F_grad_b_i.copy())

        grad_first_W, grad_first_b, _ = self.hidden_layer_grad(X_list[0], self.weights[0], self.biases[0], v_i)
        weight_grads.append(grad_first_W.copy())
        bias_grads.append(grad_first_b.copy())
        return list(reversed(weight_grads)), list(reversed(bias_grads))

    def update_thetas(self, W_grad_list, bias_grad_list, learning_rate):
        self.weights[0] = self.weights[0] - learning_rate * W_grad_list[0]
        self.biases[0] = self.biases[0] - learning_rate * bias_grad_list[0]
        for i in range(1, len(self.weights)):
            self.weights[i][0] = self.weights[i][0] - learning_rate * W_grad_list[i][0]
            self.weights[i][1] = self.weights[i][1] - learning_rate * W_grad_list[i][1]
            self.biases[i] = self.biases[i] - learning_rate * bias_grad_list[i]
