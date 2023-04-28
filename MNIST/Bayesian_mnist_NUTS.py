import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from MCMC import MCMC
from pyro.infer import NUTS
from utils import *
from torch.utils.data import Dataset, DataLoader
from pyro.infer.autoguide import init_to_value
import os
import time
PATH = os.getcwd()

x_train = torch.load(PATH+'/mnist_data/train_data.pt')
y_train = torch.load(PATH+'/mnist_data/train_labels.pt')

L = 400
class BNN(PyroModule):
  def __init__(self):
    super().__init__()

    # Hidden layer 1
    self.hidden1 = PyroModule[nn.Linear](28*28, L)
    # Weight priors
    self.hidden1.lambda_weight = PyroSample(dist.HalfCauchy(1.).expand([L, 28*28]).to_event(2))
    self.hidden1.tau_weight = PyroSample(dist.HalfCauchy(1))
    self.hidden1.weight = PyroSample(
      lambda hidden1: dist.Normal(0., (hidden1.lambda_weight) * (hidden1.tau_weight)).expand([L, 28*28]).to_event(2))
    # Bias priors
    self.hidden1.lambda_bias = PyroSample(dist.HalfCauchy(1.).expand([L]).to_event(1))
    self.hidden1.tau_bias = PyroSample(dist.HalfCauchy(1))
    self.hidden1.bias = PyroSample(
      lambda hidden1: dist.Normal(0., (hidden1.lambda_bias) * (hidden1.tau_bias)).expand([L]).to_event(1))

    # Hidden layer 2
    self.hidden2 = PyroModule[nn.Linear](L, L)
    # Weight priors
    self.hidden2.lambda_weight = PyroSample(dist.HalfCauchy(1.).expand([L, L]).to_event(2))
    self.hidden2.tau_weight = PyroSample(dist.HalfCauchy(1))
    self.hidden2.weight = PyroSample(
        lambda hidden2: dist.Normal(0., (hidden2.lambda_weight) * (hidden2.tau_weight)).expand([L, L]).to_event(2))
    # Bias priors
    self.hidden2.lambda_bias = PyroSample(dist.HalfCauchy(1.).expand([L]).to_event(1))
    self.hidden2.tau_bias = PyroSample(dist.HalfCauchy(1))
    self.hidden2.bias = PyroSample(
        lambda hidden2: dist.Normal(0., (hidden2.lambda_bias) * (hidden2.tau_bias)).expand([L]).to_event(1))

    # Output layer
    self.out = PyroModule[nn.Linear](L, 10)
    # Weight priors
    self.out.lambda_weight = PyroSample(dist.HalfCauchy(1.).expand([10, L]).to_event(2))
    self.out.tau_weight = PyroSample(dist.HalfCauchy(1))
    self.out.weight = PyroSample(
      lambda out: dist.Normal(0., (out.lambda_weight) * (out.tau_weight)).expand([10, L]).to_event(2))
    # Bias priors
    self.out.lambda_bias = PyroSample(dist.HalfCauchy(1.).expand([10]).to_event(1))
    self.out.tau_bias = PyroSample(dist.HalfCauchy(1))
    self.out.bias = PyroSample(lambda out: dist.Normal(0., (out.lambda_bias) * (out.tau_bias)).expand([10]).to_event(1))

  def forward(self, x, y=None):
      x = self.hidden1(x)
      x = F.relu(x)
      x = self.hidden2(x)
      x = F.relu(x)
      x = self.out(x)
      output = F.log_softmax(x, dim=-1)
      # Likelihood
      with pyro.plate("data", x.shape[0]):
        obs = pyro.sample("obs", dist.Categorical(probs=torch.exp(output)), obs=y)
      return torch.exp(output)

bnn = BNN()
nuts_kernel = NUTS(bnn)
mcmc_nuts = MCMC(nuts_kernel, num_samples=1, warmup_steps=1)
mcmc_nuts.run(x_train, y_train)

# Set model to evaluation mode
bnn.eval()

# Set up predictive distributions
predictive_NUTS = pyro.infer.Predictive(bnn, posterior_samples = mcmc_nuts.get_samples())

## Training predictions
predictions = predictive_NUTS(x_train)['obs'][0]
torch.save(predictions, PATH+'/mnist_data/predictions_Nuts.pt')
print("Predictions ", predictions)
print("True ", y_train)
#y_train_pred_NUTS = torch.mean(predictive_NUTS(x_train)['obs'], dim=0)
