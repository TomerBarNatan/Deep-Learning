import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

SEQ_LEN = 1007


class Snp500Dataset(Dataset):
    """Class to hold the S&P500 data as a Dataset object, so we can load it using Dataloader with torch"""

    def __init__(self, stocks, labels, transform=None):
        """
        Constructor
        :param stocks: Tensor which holds the stocks highs
        :param labels: Tensor which holds the stocks labels (i.e. next day high)
        :param transform: Transform to use for the data
        """
        self.stocks = stocks
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.stocks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        stocks = self.stocks[idx, :]
        labels = self.labels[idx, :]
        if self.transform:
            stocks = self.transform(stocks)
            labels = self.transform(labels)
        return stocks, labels


def build_sequences(stock_df):
    """
    Builds stock high values sequences from the stocks' data frame. Filter stocks with missing data.
    :param stock_df: the stocks' data frame
    :return: filtered stocks' sequences, filtered stocks' names list
    """
    names = stock_df["symbol"].unique()
    dates = stock_df["date"].unique()
    stock_sequences = np.zeros((len(names), len(dates)))
    names_filtered = []
    for i in range(len(names)):
        one_name_df = stock_df.loc[stock_df["symbol"] == names[i]]["high"].values
        if one_name_df.shape[0] == len(dates) and np.isnan(one_name_df).sum() == 0:
            stock_sequences[i, :] = one_name_df.copy()
            names_filtered.append(names[i])
    zero_indexes = np.argwhere(np.all(stock_sequences[:, ...] == 0, axis=1))
    stock_sequences = np.delete(stock_sequences, zero_indexes, axis=0)
    return stock_sequences, names_filtered


def get_amazon_google_plot(stocks_df):
    """
    Plots Amazon and Google stocks high values
    :param stocks_df: the stocks' data frame
    """
    amazon_df = stocks_df.loc[stocks_df["symbol"] == "AMZN"]
    google_df = stocks_df.loc[stocks_df["symbol"] == "GOOGL"]
    fig, ax = plt.subplots()
    plt.title("Daily max value per day")
    amazon_df.plot("date", "high", ax=ax)
    google_df.plot("date", "high", ax=ax)
    plt.legend(["Amazon", "Google"])
    plt.xlabel("Date")
    plt.ylabel("Max value")
    plt.show()
