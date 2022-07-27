import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from encoder_decoder import EncoderDecoder
import pandas as pd
import os
from pathlib import Path
from snp500_dataset import *
from utils import plot_sequence, get_device, plot_loss, Normalizer


def train(train_loader, validation_loader, lr, hidden_state, gradient_clip=1, epochs=3):
    encoder_decoder = EncoderDecoder(1, hidden_state, 1, is_prediction=True, labels_num=1)
    validation_losses = []
    optimizer = optim.Adam(encoder_decoder.parameters(), lr)
    best_loss, min_loss = float("inf"), float("inf")
    for epoch in range(epochs):
        accum_loss = 0
        for i, (batch, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(get_device())
            labels = labels.to(get_device())
            data_sequence = batch.view(batch.shape[0], batch.shape[1], 1)
            output, preds = encoder_decoder(data_sequence)
            loss_obj = encoder_decoder.loss(output, data_sequence, true_labes=labels, predicted_labels=preds)
            loss = loss_obj.item()
            accum_loss += loss
            loss_obj.backward()
            if gradient_clip:
                nn.utils.clip_grad_norm_(encoder_decoder.parameters(), max_norm=gradient_clip)
            optimizer.step()

        avg_loss = accum_loss / len(train_loader)
        best_loss = min(best_loss, avg_loss)

        validation_loss = evaluate(encoder_decoder, validation_loader)
        validation_losses.append(validation_loss)
        if epoch % 5 == 0:
            # save model till now
            results_path = f'{os.getcwd()}/saved_models/ae_snp500_prediction_lr={lr}_hidden_size={hidden_state}_' \
                           f'_gradient_clipping={gradient_clip}_'
            Path(results_path).mkdir(parents=True, exist_ok=True)
            torch.save(encoder_decoder, results_path + f"/epoch={epoch}_bestloss={best_loss}.pt")
            plot_loss(validation_losses, epochs, lr, gradient_clip, hidden_state, "validation", "snp500")

    plot_loss(validation_losses, epochs, lr, gradient_clip, hidden_state, "validation", "snp500")


def evaluate(model, validation_loader):
    model.eval()
    accum_loss = 0
    for i, (batch, labels) in enumerate(validation_loader):
        batch = batch.to(get_device())
        labels = labels.to(get_device())
        data_sequence = batch.view(batch.shape[0], batch.shape[1], 1)
        output, preds = model(data_sequence)
        loss_obj = model.loss(output, data_sequence, true_labes=labels, predicted_labels=preds)
        accum_loss += loss_obj.item()
    model.train()
    validation_loss = accum_loss / len(validation_loader.dataset)
    return validation_loss


def multi_step_prediction(model, sequence, normilizer, dates, stock_names):
    model.eval()
    time_frame = int(sequence.shape[1] / 2)
    prediction_sequence = torch.zeros(sequence.shape)
    prediction_sequence[:, :time_frame, :] = sequence[:, :time_frame, :]
    for i in range(time_frame):
        print(i)
        sequence_interval = prediction_sequence[:, i:i + time_frame, :]
        _, preds = model(sequence_interval)
        next_day = preds[:, -1, :]
        prediction_sequence[:, i + time_frame, :] = next_day
    prediction_sequence = normilizer.undo_normalize(prediction_sequence)
    sequence = normilizer.undo_normalize(sequence)
    plot_stock(sequence[0:1], prediction_sequence[0:1], dates, f"multistep prediction", f"{stock_names[0]} Multistep Prediction")


def step_prediction(model, sequence, normilizer, dates, stock_names):
    model.eval()
    reconstructed, preds = model(sequence)
    preds = normilizer.undo_normalize(preds)
    reconstructed = normilizer.undo_normalize(reconstructed)
    sequence = normilizer.undo_normalize(sequence)
    plot_stock(sequence[0:1], preds[0:1], dates, f"one step prediction", f"{stock_names[0]} One Step Prediction")
    plot_stock(sequence[0:1], preds[0:1], dates, f"reconstructed", f"{stock_names[0]} Reconstruction")


def plot_stock(original, reconstructed, dates, legend, title):
    plt.plot(dates, original[0,:])
    plt.plot(dates, reconstructed[0,:])
    plt.xlabel("date")
    plt.ylabel("stocks high")
    plt.legend(["original stock", legend])
    plt.title(title)
    plt.xticks(range(0, len(dates), 250))
    plt.show()


def grid_search(train_loader, validation_loader, learning_rates, gradient_clips, hidden_sizes):
    for lr in learning_rates:
        for gradient_clip in gradient_clips:
            for hidden_state in hidden_sizes:
                train(train_loader, validation_loader, lr, hidden_state, gradient_clip)


def get_data_loader(data, get_loader=True, batch_size=None):
    normalizer = Normalizer()
    normal_data = normalizer.normalize(data)[:, :-1]
    labels_data = normalizer.normalize(data)[:, 1:]
    tensor_data = torch.tensor(normal_data, dtype=torch.float32).view(len(normal_data), 1006, 1)
    tensor_labels = torch.tensor(labels_data, dtype=torch.float32).view(len(labels_data), 1006, 1)
    if not get_loader:
        return tensor_data, tensor_labels, normalizer
    data_set = Snp500Dataset(tensor_data, tensor_labels)
    data_loader = DataLoader(data_set, shuffle=True, batch_size=batch_size)
    return data_loader


def split_data(data_sequences, stocks_names):
    num_of_seq = len(stocks_names)
    train_data = data_sequences[:int(num_of_seq * 0.6)]
    train_names = stocks_names[:int(num_of_seq * 0.6)]

    validation_data = data_sequences[int(num_of_seq * 0.6):int(num_of_seq * 0.8)]
    validation_names = stocks_names[int(num_of_seq * 0.6):int(num_of_seq * 0.8)]

    test_data = data_sequences[int(num_of_seq * 0.8):]
    test_names = stocks_names[int(num_of_seq * 0.8):]
    return train_data, train_names, validation_data, validation_names, test_data, test_names


if __name__ == '__main__':
    stocks_df = pd.read_csv(os.getcwd() + "/data/SP 500 Stock Prices 2014-2017.csv")
    data_sequences, stocks_names = build_sequences(stocks_df)
    get_amazon_google_plot(stocks_df)
    train_data, train_names, validation_data, validation_names, test_data, test_names = split_data(data_sequences,
                                                                                                   stocks_names)
    # train_data_loader = get_data_loader(train_data)
    # validation_data_loader = get_data_loader(validation_data)
    # test_data_loader = get_data_loader(test_data)




    model = torch.load(os.getcwd() + "//trained_models//snp500_prediction_3000_epochs.pt",
                       map_location=torch.device('cpu'))
    test_tensor, _, normilizer = get_data_loader(test_data, False)
    step_prediction(model, test_tensor, normilizer, stocks_df["date"].unique()[1:], test_names)
    multi_step_prediction(model, test_tensor, normilizer, stocks_df["date"].unique()[1:], test_names)
