"""
Implementation of a feed-forward network (multi-layer perceptron, or mlp).

author: Geoffroy de Gournay
date: April 27, 2022
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .data import get_path_file, get_hyperparameters
from pickle import dump, load


class Mlp(nn.Module):
    """
    Class for implementing the neural network

    Parameters
    ----------
    n_layers: int
        number of layers
    hidden_dim: int
        hidden dimension
    n_classes: int
        number of classes
    input_dim: int
        dimension of the features
    batch_size: int
        Number of examples per batch. Final batches can have fewer examples, depending on the total number of examples
        in the dataset.
    epochs: int
        Number of training iterations over the dataset.
    learning_rate : float
        Learning rate for the optimizer.
    hyper_tuning: bool
        Indicates if the model is used for hyperparameters tuning. In this case, the model doesn't print the losses
        during training
    use_saved_hyper_params: bool
        Indicates if hyperparameters need to be loaded from yaml file
    """

    def __init__(self, n_layers: int = 2,
                 hidden_dim: int = 50,
                 n_classes: int = 2,
                 input_dim: int = 108,
                 batch_size: int = 1028,
                 epochs: int = 200,
                 learning_rate: float = 0.001,
                 dropout_rate: float = 0.5,
                 hyper_tuning: bool = False,
                 use_saved_hyper_params: bool = False
                 ):
        super(Mlp, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()  # combines nn.LogSoftmax and nn.NLLLoss
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = epochs
        self.hyper_tuning = hyper_tuning
        if use_saved_hyper_params:
            params = get_hyperparameters()['parameters']
            self.batch_size = params['batch_size']
            self.dropout_rate = params['dropout_rate']
            hidden_dim = params['hidden_dim']
            self.learning_rate = params['learning_rate']
            n_layers = params['n_layers']
        else:
            self.batch_size = batch_size
            self.dropout_rate = dropout_rate
            self.learning_rate = learning_rate
        self.network = build_mlp(input_size=input_dim,
                                 output_size=n_classes,
                                 n_layers=n_layers,
                                 hidden_size=hidden_dim,
                                 dropout_rate=dropout_rate)
        self.optimizer = torch.optim.Adam(self.network.parameters())
        self.network.to(self.device)
        # Calibrated preprocessing tools that were used for training an that are necessary for inference
        self.encoder = None  # sklearn.preprocessing.OneHotEncoder
        self.lb = None  # sklearn.preprocessing.LabelBinarizer
        self.scaler = None  # sklearn.preprocessing.StandardScaler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pytorch forward method used to perform a forward pass of inputs through the network
        :param x: features observed (shape [batch size, dim(features)])
        :return:
            output (torch.Tensor): networks predicted income class for a given observation (shape [batch size])
        """
        logits = self.network(x)
        return logits

    def train_model(self, x_train: np.array, y_train: np.array):
        """
        Train the NN
        :param x_train:
        :param y_train:
        :return:
        """
        data_set = CensusDataset(x_train, y_train, self.device)
        train_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True)
        self.network.train()

        for epoch in range(self.epochs):
            last_loss = 0
            running_loss = 0
            for i, data in enumerate(train_loader):
                n_batch = len(train_loader)  # number of batches
                # Every data instance is an input + label pair
                inputs, labels = data

                # Zero your gradients for every batch!
                self.optimizer.zero_grad()

                # Make predictions for this batch
                outputs = self.network(inputs)

                # Compute the loss and its gradients
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                if epoch % 20 == 0 and i % n_batch == n_batch - 1 and not self.hyper_tuning:
                    last_loss = running_loss / n_batch  # loss per batch
                    print('epoch {} loss: {}'.format(epoch, last_loss))
                    running_loss = 0.

        return last_loss

    def predict(self, x: np.array) -> np.array:
        """
        Use model for inference
        :param x:
        :return:
        """
        self.network.eval()
        x = torch.from_numpy(x).to(self.device).to(torch.float32)
        logits = self.network(x)
        y_pred = logits.argmax(dim=1).cpu().numpy().reshape(-1)
        return y_pred

    def save_model(self, encoder, lb, scaler):
        """
        Save the model and the preprocessing tools used to calibrate the model
        :param encoder:
        :param lb:
        :param scaler:
        :return: (OneHotEncoder, LabelBinarizer, StandardScaler)
        """
        model_path = get_path_file('model/mlp.pt')
        torch.save(self.state_dict(), model_path)
        # save the tools used for pre-processing data
        dump(encoder, open('model/encoder.pkl', 'wb'))
        dump(lb, open('model/lb.pkl', 'wb'))
        dump(scaler, open('model/scaler.pkl', 'wb'))

    def load_model(self):
        """
        load model and pre-processing tools needed for inference
        :return:
        """
        model_path = get_path_file('model/mlp.pt')
        self.load_state_dict(torch.load(model_path))
        self.encoder = load(open('model/encoder.pkl', 'rb'))
        self.lb = load(open('model/lb.pkl', 'rb'))
        self.scaler = load(open('model/scaler.pkl', 'rb'))


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        hidden_size: int,
        dropout_rate) -> nn.Module:
    """
    Builds a multi-layer perceptron in Pytorch based on a user's input
    :param input_size: the dimension of inputs to be given to the network
    :param output_size: the dimension of the output
    :param n_layers: t the number of hidden layers of the network
    :param hidden_size: the size of each hidden layer
    :param dropout_rate: During training, probability of the elements of the input tensor being set to 0.
    :return:
    An instance of (a subclass of) nn.Module representing the network.
    """
    # seequence of  affine operations: y = Wx + b followed by RelU activation.
    layer_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]

    for layer in range(n_layers):
        layer_list.append(nn.Linear(hidden_size, hidden_size))
        layer_list.append(nn.Dropout(p=dropout_rate))
        layer_list.append(nn.ReLU())
    layer_list.append(nn.Linear(hidden_size, output_size))
    mlp = nn.Sequential(*layer_list)
    return mlp


class CensusDataset(Dataset):
    def __init__(self, features: np.array, labels: np.array, device: str):
        self.features = torch.from_numpy(features).to(device).to(torch.float32)
        self.labels = torch.from_numpy(labels).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
