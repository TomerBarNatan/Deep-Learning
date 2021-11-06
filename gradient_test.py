import numpy as np
import matplotlib.pyplot as plt
from softmax import softmax_regression, softmax_grad
from scipy.io import loadmat


iter = 10
eps = [np.power(0.5, i) for i in range(iter)]
PeaksData = loadmat('PeaksData.mat')
X = PeaksData.get("Yt")
X = X[:,:100]
C = PeaksData.get("Ct")
C = C[:,:100]


def grad_test(X, W, w_p, bias, c_p, C):
	d = np.random.rand(W.shape[0], 1)
	d /= np.linalg.norm(d)
	F_0 = softmax_regression(X, W, bias, C)
	grad_0 = softmax_grad(X, W, w_p, c_p, bias)
	y_0 = []
	y_1 = []
	for epsilon in eps:
		F_i = softmax_regression(X, W + epsilon*d, bias, C)
		F_1 = F_0 + epsilon * d.T @ grad_0
		y_0.append(abs(F_i - F_0))
		y_1.append(abs(F_i - F_1))
	return y_0, y_1


def draw_results(y_0, y_1):
	plt.semilogy(range(iter), y_0)
	plt.semilogy(range(iter), y_1)
	plt.legend("Zero order approx", "First order approx")
	plt.title("Successful grad test in semilog scale plot")
	plt.xlabel("iteration")
	plt.ylabel("error")
	plt.plot()


W_0 = np.random.rand(2, 100)
w_p = (W_0.T[0]).T
c_p = (C.T[0]).T

res_0, res_1 = grad_test(X, W_0, w_p, 0, c_p, C)
draw_results(res_0, res_1)
