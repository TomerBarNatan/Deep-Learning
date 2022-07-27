import torch
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_sequence(original, reconstructed, epochs=0, lr=None, gradient_clip=None, hidden_sate=None, dataset="",
                  num_of_plots=2, path=None, title=None, legend=None):
    """
    Plots original sequence and its AE reconstruction
    :param original: original sequence
    :param reconstructed: reconstructed sequence
    :param epochs: number of epochs (used for path saving purpose only)
    :param lr: learning rate (used for path saving purpose only)
    :param gradient_clip: gradient clip value (used for path saving purpose only)
    :param hidden_sate: hidden state size (used for path saving purpose only)
    :param dataset: data set name (used for path saving purpose only)
    :param num_of_plots: used to plot more than 1 plot for method comparison
    :param path: the path where to plot should be saved (None by default and saved according to given input)
    :return: Saves the plots of original sequence compared to the reconstructed in the desired path
    """
    if path:
        fig_path = f'{os.getcwd()}/{path}'
    else:
        fig_path = f'{os.getcwd()}/graphs/sequences/ae_{dataset}_lr={lr}_hidden_size={hidden_sate}_epochs={epochs}' \
                   f'_gradient_clipping={gradient_clip}_'
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    _, axis1 = plt.subplots(1, 1)
    axis1.plot(np.arange(0, original.shape[1], 1), original[0, :, :].detach().cpu().numpy())
    axis1.plot(np.arange(0, reconstructed.shape[1], 1), reconstructed[0, :, :].detach().cpu().numpy())
    axis1.set_xlabel("date")
    axis1.set_ylabel("signal value")
    axis1.legend(("original signal", legend))

    axis1.set_title(title)
    plt.savefig(fig_path + f"/{dataset}_data_recunstruct1.jpg")
    if num_of_plots > 1:
        _, axis2 = plt.subplots(1, 1)
        axis2.plot(np.arange(0, original.shape[1], 1), original[1, :, :].detach().cpu().numpy())
        axis2.plot(np.arange(0, reconstructed.shape[1], 1), reconstructed[1, :, :].detach().cpu().numpy())
        axis2.set_xlabel("time")
        axis2.set_ylabel("signal value")
        axis2.legend(("original signal 1", "reconstructed signal 1"))
        axis2.set_title(f"{dataset} Signal Graph Reconstruction")
        plt.savefig(fig_path + f"/{dataset}_data_recunstruct2.jpg")


def plot_loss(losses, epochs, lr, gradient_clip, hidden_sate, data_kind="validation", dataset='', path=None):
    """
    Plots the loss according to the given params
    :param losses: list of loss values
    :param epochs: number of epochs (size of losses list should be the same)
    :param lr: learning rate used (for saving the plot purpose only)
    :param gradient_clip: gradient clip value (used for path saving purpose only)
    :param hidden_sate: hidden state size (used for path saving purpose only)
    :param data_kind: should be either train or validation according to the given loss
    :param dataset: name of the data set used
    :param path: the desired path to save the plot
    :return: Saves the loss plot in the desired path
    """
    if path:
        fig_path = f"{os.getcwd()}/{path}"
    else:
        fig_path = f'{os.getcwd()}/graphs/{data_kind}/ae_loss_{dataset}_lr={lr}_hidden_size={hidden_sate}_epochs={epochs}' \
                   f'_gradient_clipping={gradient_clip}_'
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    _, axis1 = plt.subplots(1, 1)
    axis1.plot(np.arange(1, len(losses) + 1, 1), losses)
    axis1.set_xlabel("epochs")
    axis1.set_ylabel(f"{data_kind} loss")
    axis1.set_title(f"{data_kind} loss")
    plt.savefig(fig_path + f"/{dataset}_data_loss_{data_kind}.jpg")


def plot_accuracy(accuracies, epochs, lr, gradient_clip, hidden_sate, data_kind="validation", dataset='', path=None):
    """
    Plots the accuracy percentage
    :param accuracies: list of accuracy values
    :param epochs: number of epochs (size of losses list should be the same)
    :param lr: learning rate used (for saving the plot purpose only)
    :param gradient_clip: gradient clip value (used for path saving purpose only)
    :param hidden_sate: hidden state size (used for path saving purpose only)
    :param data_kind: data_kind: should be either train or validation according to the given loss
    :param dataset: name of the data set used
    :param path: the desired path to save the plot
    :return: Saves the accuracy plot in the desired path
    """
    if path:
        fig_path = f"{os.getcwd()}/{path}"
    else:
        fig_path = f'{os.getcwd()}/graphs/{data_kind}/ae_accuracy_{dataset}_lr={lr}_hidden_size={hidden_sate}_epochs={epochs}' \
                   f'_gradient_clipping={gradient_clip}_'
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    _, axis1 = plt.subplots(1, 1)
    axis1.plot(np.arange(1, len(accuracies) + 1, 1), accuracies)
    axis1.set_xlabel("epochs")
    axis1.set_ylabel(f"{data_kind} accuracy")
    axis1.set_title(f"{data_kind} accuracy")
    plt.savefig(fig_path + f"/{dataset}_accuracy_{data_kind}.jpg")


def get_device():
    """
    Determines the device to use during the execution
    :return: 'cuda' (GPU) if its available, otherwise returns 'cpu'
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_optimizer(optimizer_name, model_params, lr):
    """
    Gets the desired optimizer (according to the given name) from torch.optim built with the given params
    :param optimizer_name: name of the desired optimizer (case sensitive)
    :param model_params: the model parameters
    :param lr: learning rate value
    :return: the desired optimizer (from torch.optim) with the given parameters
    """
    optimizers = {
        'Adadelta': torch.optim.Adadelta,
        'Adagrad': torch.optim.Adagrad,
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SparseAdam': torch.optim.SparseAdam,
        'Adamax': torch.optim.Adamax,
        'ASGD': torch.optim.ASGD,
        'LBFGS': torch.optim.LBFGS,
        'NAdam': torch.optim.NAdam,
        'RAdam': torch.optim.RAdam,
        'RMSprop': torch.optim.RMSprop,
        'Rprop': torch.optim.Rprop,
        'SGD': torch.optim.SGD,
    }
    selected_optimizer = (optimizers.get(optimizer_name))(model_params, lr) if optimizer_name in optimizers.keys() \
        else (optimizers.get('Adam'))(model_params, lr)
    return selected_optimizer


class Normalizer:
    """
    A class which helps us to normalize and un-normalize the stocks data for our model
    """

    def __init__(self):
        self.max_stock_values = None
        self.min_stock_values = None
        self.mean_stock_values = None

    def normalize(self, stock_sequences):
        """
           :param stock_sequences = np.array of shape (num of sequences X sequence length)
           normalizes each sequence to range of [0,1] with mean=0.5
        """
        self.max_stock_values = stock_sequences.max(axis=1).reshape(-1, 1)
        self.min_stock_values = stock_sequences.min(axis=1).reshape(-1, 1)
        stock_sequences = (stock_sequences - self.min_stock_values) / (self.max_stock_values - self.min_stock_values)
        self.mean_stock_values = stock_sequences.mean(axis=1).reshape(-1, 1)
        normalized_stock_sequence = stock_sequences / (2 * self.mean_stock_values)

        return normalized_stock_sequence

    def undo_normalize(self, stock_sequences):
        """
        :param stock_sequences: batch of stocks sized (batch_size X sequence_length)
        """
        batch_size, T, _ = stock_sequences.shape
        assert self.mean_stock_values is not None, "must normalize before undo normalize"
        stock_sequences = stock_sequences.view(batch_size, T).detach().numpy() * (
                    2 * self.mean_stock_values[:batch_size, :])
        stock_sequences_unnormalized = (stock_sequences * (
                self.max_stock_values[:batch_size, :] - self.min_stock_values[:batch_size, :])) + self.min_stock_values[
                                                                                                  :batch_size, :]
        return stock_sequences_unnormalized
