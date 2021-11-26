import numpy as np


def tanh():
	return np.tanh


def tanh_grad(x):
	return np.ones(x.shape) - (np.tanh(x)) ** 2


def ReLU(x):
	return np.maximum(0, x)


def ReLU_grad(x):
	vfunc = np.vectorize(lambda t: 1 if t >= 0 else 0)
	return vfunc(x)
