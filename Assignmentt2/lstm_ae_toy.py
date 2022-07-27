import torch.utils.data
import torch.nn as nn
from synthetic_dataset import create_synthetic_dataset
from encoder_decoder import EncoderDecoder
import os
from pathlib import Path
from utils import plot_sequence, plot_loss, get_device, get_optimizer


def train(train_loader, validation_set, lr, hidden_state, batch_size, gradient_clip, optimizer_name, epochs=1000):
    encoder_decoder = EncoderDecoder(1, hidden_state, 1)
    encoder_decoder = encoder_decoder.to(get_device())
    validation_set = validation_set.to(get_device())
    mse = nn.MSELoss()
    mse = mse.to(get_device())
    validation_losses = []
    train_losses = []
    optimizer = get_optimizer(optimizer_name, encoder_decoder.parameters(), lr)
    best_train_loss, best_validation_loss = float("inf"), float("inf")
    min_validation_reconstruct = None
    for epoch in range(epochs):
        accum_loss = 0
        print(epoch)
        for i, batch in enumerate(train_loader, 0):
            batch = batch.to(get_device())
            optimizer.zero_grad()
            output, _ = encoder_decoder(batch)
            loss_obj = mse(output, batch)
            loss = loss_obj.item()
            accum_loss += loss
            loss_obj.backward()
            if gradient_clip:
                nn.utils.clip_grad_norm_(encoder_decoder.parameters(), max_norm=gradient_clip)
            optimizer.step()

        avg_loss = accum_loss / len(train_loader)
        train_losses.append(avg_loss)
        best_train_loss = min(best_train_loss, avg_loss)

        validation_reconstruct, validation_loss = evaluate(encoder_decoder, validation_set, mse)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            min_validation_reconstruct = validation_reconstruct
            best_validation_loss = validation_loss

        if epoch % 20 == 0:
            # save model till now
            results_path = f'{os.getcwd()}/saved_models/ae_toy_{optimizer_name}_lr={lr}_hidden_size={hidden_state}_' \
                           f'_gradient_clipping={gradient_clip}_batchsize={batch_size}'
            Path(results_path).mkdir(parents=True, exist_ok=True)
            torch.save(encoder_decoder, results_path + f"/epoch={epoch}_bestloss={best_validation_loss}.pt")

    plot_loss(train_losses, epochs, lr, gradient_clip, hidden_state, "train", "mnist")
    plot_loss(validation_losses, epochs, lr, gradient_clip, hidden_state,"validation", "mnist")
    plot_sequence(validation_set, min_validation_reconstruct, epochs, lr, gradient_clip, hidden_state,"mnist",2)


def evaluate(encoder_decoder, validation_set, mse):
    encoder_decoder.eval()
    mse.eval()
    reconstruct_validation, _ = encoder_decoder(validation_set)
    validation_loss = mse(reconstruct_validation, validation_set).item()
    encoder_decoder.train()
    mse.train()
    return reconstruct_validation, validation_loss


if __name__ == '__main__':
    train_data, validation, test = create_synthetic_dataset()

    batch_sizes = [32, 64]
    learning_rates = [0.001, 0.01]
    gradient_clips = [1, 0]
    hidden_state_sizes = [40, 25, 10]
    # grid search
    for batch_size in batch_sizes:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        for lr in learning_rates:
            for gradient_clip in gradient_clips:
                for hidden_state in hidden_state_sizes:
                    print(f'Executing params:\nbatch size: {batch_size}\nlr: {lr}\ngradient_clip: {gradient_clip}'
                          f'\nhidden state: {hidden_state}\n')
                    train(train_loader, validation, lr, hidden_state, batch_size, gradient_clip, "Adam")
