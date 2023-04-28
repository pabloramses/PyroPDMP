import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision import models
import time
import os
import copy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25):
    """
    Function used to train and validate the different models. It takes as inputs
    INPUT:
    model: a torch.nn.model instance to train and validate.
    criterion: a loss function from the nn module of torch
    optmizer: a tuned optimizer from the nn module of torch
    num_epochs: the number of times the model is expected to see the complete training set to update parameters.
    train_loader: a dataloader instance of the dataloader class from torch, that contains batched data to be used for training
    val_loader: a dataloader instance of the dataloader class from torch, that contains batched data to be used for validation

    OUTPUT:
    model: the model that received tuned according to the optimal validation setting
    train_tracker: a list with the running loss over epochs on the training set.
    val_tracker: a list with the running loss over epochs on the validation set.

    NOTE: this function was taken from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html and adapted to the current structure.
    """
    since = time.time()
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 2000

    train_tracker = []
    val_tracker = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            if phase == 'train':
                dataloader = train_loader
            else:
                dataloader = val_loader

            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val':
                val_tracker.append(epoch_loss)
            else:
                train_tracker.append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best train Loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    return model, train_tracker, val_tracker


def test_model(model, test_loader):
    """
    Function to iterate over a test_loader and compute the output of a given model

    INPUT
    model: an instance of any model class from torch
    test_loader: an instance of the dataloader class from torch containing the data to be estimated

    OUTPUT
    output: a numpy array containing the estimations for each sample in the test_loader
    """

    model.to(device)  # move the model to the GPU (or CPU) if it was not already.
    model.eval()  # Set model to evaluate mode, so that weights are not modified
    outputs = []
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        # optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs.append(model(inputs))
    cond = 0
    for i in outputs:
        if cond == 0:
            output = i.cpu().detach().numpy()
            cond += 1
        else:
            output = np.concatenate((output, i.cpu().detach().numpy()))

    return output


class MLP(torch.nn.Module):

    def __init__(self, U):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(28*28, U)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(U, U)
        self.linear3 = torch.nn.Linear(U, 10)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x