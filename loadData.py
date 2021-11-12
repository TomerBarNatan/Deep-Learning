import numpy as np
from scipy.io import loadmat


def extract_grad_test_data(data_set, number_of_batches):
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


def extract_sgd_data():
	number_of_butches = 200
	learningRate = 0.0001
	iterations = 800
	PeaksData = loadmat('PeaksData.mat')
	trainSet = np.array(PeaksData['Yt'])
	trainSetLabels = np.array(PeaksData['Ct'])
	validationSet = np.array(PeaksData['Yv'])
	validationSetLabels = np.array(PeaksData['Cv'])
	testSetSize = 200
	# shuffle validation set
	idx = np.random.permutation(len(validationSetLabels[0]))
	testSetX = validationSet[:, idx][:, :testSetSize]
	testSetY = validationSetLabels[:, idx][:, :testSetSize]
	# shuffle training set
	idx = np.random.permutation(len(trainSetLabels[0]))
	trainSetX = trainSet[:, idx]
	trainSetY = trainSetLabels[:, idx]
	# split into batches
	trainSetX_batches = np.array_split(trainSetX, number_of_butches, axis=1)
	trainSetY_batches = np.array_split(trainSetY, number_of_butches, axis=1)
	# set parameters
	output_size = len(trainSetY_batches[0][:, 0])
	input_size = trainSetX_batches[0].shape[0]
	# theta = np.random.rand(input_size, output_size)
	theta = np.zeros([input_size, output_size])
	bias = np.zeros([1, output_size])
	return trainSet, trainSetLabels,validationSet, validationSetLabels, theta, bias