import numpy as np
import math


class ResNet:
    """
    ResNet is a class which represents the neural network.
    """
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

    def res_forward_step(self, x, Ws, bias):
        """
        A forward step in the residual network, alters the data in transition from layer k to layer k+1
        :param x: the data in layer k
        :param Ws: the weights in layer k (we have W1 and W2)
        :param bias: the bias in layer k
        :return: the linear result of W1@x + bias, and x + W2 @ f(linear) where f is the configured activation function
        """
        linear = Ws[0] @ x + bias
        nonlinear = x + Ws[1] @ self.activation(linear)
        return linear, nonlinear

    def forward_step(self, x, W, bias):
        """
       The first forward step of the ResNet, used to change dimensions at the beginning of the network.
       :param x: the data
       :param W: the weights
       :param bias: the bias
       :return: the linear result of W@x + bias, and f(linear) where f is the configured activation function
       """
        linear = W @ x + bias
        nonlinear = self.activation(linear)
        return linear, nonlinear

    def forward(self, x_0, C):
        """
        Forward routine of the whole residual network, starting from layer 1 and performing n forward steps (including
        the last step of the softmax function)
        :param x_0: the initial data
        :param C: the indicators
        :return: cost of the entire residual network, probabilities matrix, linear (W@x+bias) and non-linear
        (activation(linear)) results for each layer (as lists).
        """
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

    def res_hidden_layer_grad(self, X, Ws, b, v):
        """
        Calculate the gradient of an individual hidden layer.
        :param X: the current layer's data
        :param Ws: the current layer's weights (W1 and W2)
        :param b: the current layer's bias
        :param v: a vector v (from the next layer)
        :return: The hidden layer gradient w.r.t weights (W1 and W2), bias and the new v vector to use in the previous
        layer.
        """
        linear = Ws[0] @ X + b
        batch_size = linear.shape[1]
        grad_activation = self.activation_gradient(linear)
        grad_W1 = (1 / batch_size) * (grad_activation * (Ws[1].T @ v)) @ X.T
        grad_W2 = (1 / batch_size) * v @ (self.activation(linear)).T
        grad_b = (1 / batch_size) * np.sum((grad_activation * (Ws[1].T @ v)), axis=1, keepdims=True)
        grad_X = v + (Ws[0].T @ (grad_activation * (Ws[1].T @ v)))
        return grad_W1, grad_W2, grad_b, grad_X

    def hidden_layer_grad(self, X, W, b, v):
        """
        Calculate the gradient of the first layer.
        :param X: the first layer's data
        :param W: the first layer's weights
        :param b: the first layer's bias
        :param v: a vector v (from the second layer)
        :return: The first layer gradient w.r.t weights, bias and the new v vector to use in the previous layer
        """
        linear = W @ X + b
        batch_size = linear.shape[1]
        grad_activation = self.activation_gradient(linear)
        common = grad_activation * v
        grad_W = (1 / batch_size) * common @ X.T
        grad_b = (1 / batch_size) * np.sum(common, axis=1, keepdims=True)
        grad_X = W.T @ common
        return grad_W, grad_b, grad_X

    def backpropagation(self, X_list, C):
        """
        The backpropagation process of the network.
        :param X_list: a list of the data X in each layer
        :param C: the indicators
        :return: gradients list of each layer w.r.t weights and bias
        """
        layer_number = len(X_list)
        x_grads = []
        weight_grads = []
        bias_grads = []

        # last layer gradient
        W_grad, b_grad, x_grad = self.backward_last_layer(X_list, C)
        x_grads.insert(0, x_grad.copy())
        weight_grads.insert(0, W_grad.copy())
        bias_grads.insert(0, b_grad.copy())
        v_i = x_grad.copy()

        # hidden layer grads
        for i in range(layer_number - 2, 1, -1):
            F_grad_W1_i, F_grad_W2_i, F_grad_b_i, v_i = self.backward_hidden_layer(X_list, i, v_i)
            x_grads.insert(0, v_i)
            weight_grads.insert(0, [F_grad_W1_i.copy(), F_grad_W2_i.copy()])
            bias_grads.insert(0, F_grad_b_i.copy())

        grad_first_W, grad_first_b, grad_first_X = self.hidden_layer_grad(X_list[0], self.weights[0], self.biases[0], v_i)
        x_grads.insert(0, grad_first_X)
        weight_grads.insert(0, grad_first_W.copy())
        bias_grads.insert(0, grad_first_b.copy())
        return x_grads, weight_grads, bias_grads

    def backward_last_layer(self, X_list, C):
        """
        Calculate the gradient of the last layer.
        :param X_list: all nonlinear x's collected at forward steps
        :param C: an indicator matrix for labeling
        :return: The last layer gradient w.r.t weights, bias and x
        """
        W_grad, b_grad, x_grad = self.softmax_gradient(X_list[-1], self.weights[-1], C, X_list[-2])
        return W_grad, b_grad, x_grad

    def backward_hidden_layer(self, X_list, i, v):
        """
        Backward step for the first hidden layer
        :param X_list: list of X (data) of each layer (from the forward pass)
        :param i: index of the current layer
        :param v: vector v from the previous backward step
        :return: New vector v for the next backward step to use
        """
        F_grad_W1_i, F_grad_W2_i, F_grad_b_i, grad_X_i = self.res_hidden_layer_grad(X_list[i - 1], self.weights[i - 1],
                                                                               self.biases[i - 1],
                                                                               v)
        return F_grad_W1_i, F_grad_W2_i, F_grad_b_i, grad_X_i

    def update_thetas(self, W_grad_list, bias_grad_list, learning_rate):
        """
        Update the weights (W1 and W2) and biases of the network
        :param W_grad_list: list of gradients w.r.t weights
        :param bias_grad_list: list of gradients w.r.t bias
        :param learning_rate: the learning rate
        """
        self.weights[0] = self.weights[0] - learning_rate * W_grad_list[0]
        self.biases[0] = self.biases[0] - learning_rate * bias_grad_list[0]
        for i in range(1, len(self.weights)):
            self.weights[i][0] = self.weights[i][0] - learning_rate * W_grad_list[i][0]
            self.weights[i][1] = self.weights[i][1] - learning_rate * W_grad_list[i][1]
            self.biases[i] = self.biases[i] - learning_rate * bias_grad_list[i]
