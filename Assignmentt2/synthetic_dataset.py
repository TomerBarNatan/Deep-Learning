import random
import matplotlib.pyplot as plt
import torch

NUM_OF_SEQUENCE = 10000
SEQ_LEN = 50


def create_synthetic_dataset():
    """
    Generate a random synthetic data according to task requirements
    :return: train, validation and test sets of the generated synthetic data
    """
    random.seed(0)
    data = torch.FloatTensor(NUM_OF_SEQUENCE, SEQ_LEN, 1).uniform_(0, 1)
    indexes = torch.randint(high=31, low=20, size=(NUM_OF_SEQUENCE,))
    for i in range(len(indexes)):
        data[i, indexes[i] - 5:indexes[i] + 6, ] *= 0.1

    train = data[:int(NUM_OF_SEQUENCE * 0.6), :, :]
    validation = data[int(NUM_OF_SEQUENCE * 0.6):int(NUM_OF_SEQUENCE * 0.8), :, :]
    test = data[int(NUM_OF_SEQUENCE * 0.8):, :, :]
    return train, validation, test


def plot_synthetic_data(series, example_num):
    """
    Plots synthetic data example
    :param series: the synthetic data sequence to plot
    :param example_num: data index (for title formatting)
    """
    plt.plot(range(SEQ_LEN), series)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title(f"Synthetic data example {example_num}")
    plt.show()


if __name__ == '__main__':
    train, validation, test = create_synthetic_dataset()
    for i in [1, 2, 3]:
        plot_synthetic_data(train[i, :, :], example_num=i)
