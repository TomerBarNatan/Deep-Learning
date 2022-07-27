import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from encoder_decoder import EncoderDecoder
import matplotlib.pyplot as plt
from utils import get_device, plot_loss, plot_accuracy
import os
from pathlib import Path


def train(train_loader, test_loader, lr, hidden_state, batch_size, gradient_clip=1, is_classification=False,
          num_of_labels=0, epochs=1000):
    encoder_decoder = EncoderDecoder(28, hidden_state, 28, is_classification=is_classification,
                                     labels_num=num_of_labels)
    encoder_decoder = encoder_decoder.to(get_device())
    optimizer = optim.Adam(encoder_decoder.parameters(), lr)
    best_loss, min_loss = float("inf"), float("inf")
    model_path = ""
    accuracies = []
    losses = []
    for epoch in range(epochs):
        print(epoch)
        for i, (batch, labels) in enumerate(train_loader):
            batch = batch.to(get_device())
            labels = labels.to(get_device())
            data_sequence = batch.view(batch_size, 28, 28)
            optimizer.zero_grad()
            output, preds = encoder_decoder(data_sequence)
            loss_obj = encoder_decoder.loss(output, data_sequence, labels, preds)
            loss_obj.backward()
            if gradient_clip:
                nn.utils.clip_grad_norm_(encoder_decoder.parameters(), max_norm=gradient_clip)
            optimizer.step()
        if is_classification:
            accuracy, loss = get_test_results(encoder_decoder, test_loader)
            accuracies.append(accuracy)
            losses.append(loss)
        if epoch % 100 == 0:
            results_path = f'{os.getcwd()}/saved_models/ae_mnist'
            Path(results_path).mkdir(parents=True, exist_ok=True)
            model_path = results_path + f"/epoch={epoch}_bestloss={best_loss}.pt"
            torch.save(encoder_decoder, model_path)
    if is_classification:
        plot_accuracy(accuracies, epochs, lr, gradient_clip, hidden_state, "test", "mnist")
        plot_loss(losses, epochs, lr, gradient_clip, hidden_state, "test", "mnist")
    return model_path


def get_test_results(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    i = 1
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images_sequence = images.view(images.shape[0], 28, 28)
            outputs, preds = model.forward(images_sequence)
            _, predicted = torch.max(preds, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += model.loss(images, outputs, labels, preds)
            i += i
    loss = loss / i
    accuracy = (correct / total) * 100
    model.train()
    return loss, accuracy


def plot_imgs(original, reconstruct):
    plt.title('Original Image')
    plt.imshow(original)
    plt.show()

    plt.title('Reconstructed image')
    plt.imshow(reconstruct.detach().numpy())
    plt.show()


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_data = MNIST('./data', train=True, download=True, transform=transform)
    test_data = MNIST('./data', train=False, transform=transform)

    batch_size = 32
    lr = 0.001
    gradient_clip = 1
    hidden_state_size = 20
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

    # model_path = train(train_loader, test_loader, lr, hidden_state_size, batch_size, gradient_clip,
    #                    is_classification=True, num_of_labels=10)

    model_path = 'trained_models/mnist_reconstruct_1000_hidden=20_batch=32_gradientclip=1_lr=0.001.pt'

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=3, shuffle=True)
    encoder_decoder = torch.load(model_path)
    for i, (batch, labels) in enumerate(test_loader, 0):
        data_sequence = batch.view(3, 28, 28)
        output, _ = encoder_decoder(data_sequence)
        for j in range(3):
            plot_imgs(batch[j][0], output[j])
        break
