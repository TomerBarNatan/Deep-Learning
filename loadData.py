import numpy as np
from scipy.io import loadmat


def extract_data(data_set, number_of_batches):
	data = loadmat(data_set)
	train_set = np.array(data['Yt'])
	train_set_labels = np.array(data['Ct'])

	idx = np.random.permutation(len(train_set_labels[0]))
	train_set_X = train_set[:, idx]
	train_set_C = train_set_labels[:, idx]

	train_set_X_batches = np.array_split(train_set_X, number_of_batches, axis=1)
	train_set_C_batches = np.array_split(train_set_C, number_of_batches, axis=1)

	output_size = len(train_set_C_batches[0][:, 0])
	input_size = train_set_X_batches[0].shape[0]
	bias = np.zeros([1, output_size])
	W = np.random.rand(input_size, output_size)
	return train_set_X_batches, W, bias, train_set_C_batches
