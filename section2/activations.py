import numpy as np


def tanh(x):
	"""
	:return: tanh activation function
	"""
	return np.tanh(x)


def tanh_grad(x):
	"""
	tanh gradient function
	:param x: point where we calculate the gradient of tanh
	:return: gradient of tanh in point x
	"""
	return np.ones(x.shape) - (np.tanh(x)) ** 2


def ReLU(x):
	"""
	ReLU activation function
	:param x: point on which to calculate the ReLU
	:return: point/vector in which each entry is either 0 or positive
	"""
	return np.maximum(0, x)


def ReLU_grad(x):
	"""
	ReLU gradient function
	:param x: point where we calculate the gradient of ReLU
	:return: gradient of ReLU in point x
	"""
	vfunc = np.vectorize(lambda t: 1 if t >= 0 else 0)
	return vfunc(x)
