import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    A class which implements an LSTM AE, which is capable of reconstructing data, classifying data and predicting data.
    """
    def __init__(self, input_size, hidden_size, output_size, labels_num=0, is_classification=False,
                 is_prediction=False):
        """
        LSTM AE constructor
        :param input_size: input dimension
        :param hidden_size: desired hidden state size
        :param output_size: desired output dimension
        :param labels_num: number of labels
        :param is_classification: is the task a classification task
        :param is_prediction: is the task a prediction task
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cross_entropy = None
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.linear_construct_x = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.non_linear_construct = nn.Tanh()
        self.mse = nn.MSELoss()
        self.is_prediction = is_prediction
        self.is_classification = is_classification
        if self.is_classification or self.is_prediction:
            self.linear_classify = nn.Linear(self.hidden_size, labels_num)
            self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x_series):
        """
        Forward pass of the LSTM AE, which does both encoding and decoding according to the given task
        :param x_series: input data sequence
        :return: if the task is reconstruction only, returns the reconstructed data
                 if the task includes classification, returns both the reconstructed data and the classification
                 if the task includes prediction, returns both the reconstructed data and the predictions
        """
        encoder_output, _ = self.encoder.forward(x_series)
        z = encoder_output[:, -1]
        z_repeat = z.repeat(1, x_series.shape[1]).view(encoder_output.shape)
        decoded, _ = self.decoder.forward(z_repeat)
        constructed_x = self.non_linear_construct(self.linear_construct_x(decoded))
        if self.is_classification:
            h_t = decoded[:,-1]
            preds = self.linear_classify(h_t)
            return constructed_x, preds
        if self.is_prediction:
            h_t = decoded
            preds = self.linear_classify(h_t)
            return constructed_x, preds
        return constructed_x, None

    def loss(self, origin, reconstructed, true_labes=None, predicted_labels=None):
        """
        The loss function of the LSTM AE
        :param origin: original data
        :param reconstructed: reconstructed data
        :param true_labes: true data labels
        :param predicted_labels: predicted data labels
        :return: if the task is reconstruction only, returns the MSE loss of the origin data vs. the reconstructed data
                 if the task includes classification, returns the average of the cross entropy loss of the predicted
                    labels vs. the true labels and the MSE loss of the origin data vs. the reconstructed data
                 if the task includes prediction, returns the average of the MSE loss of the predicted labels vs. the
                    true labels and the MSE loss of the origin data vs. the reconstructed data
        """
        if self.is_classification:
            return (self.cross_entropy(predicted_labels, true_labes) + self.mse(origin, reconstructed)) / 2
        if self.is_prediction:
            return (self.mse(predicted_labels, true_labes) + self.mse(origin, reconstructed)) / 2
        else:
            return self.mse(origin, reconstructed)
