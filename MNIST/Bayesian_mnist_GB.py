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
from utils import *
from pyro.infer.autoguide import init_to_value
import os
import time
from Boomerang import Boomerang

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


parameters = ["hidden1.weight", "hidden1.bias", "out.weight", "out.bias", "hidden2.weight", "hidden2.bias"]
hyperparameters = ["hidden1.lambda_weight", "hidden1.tau_weight", "hidden1.lambda_bias", "hidden1.tau_bias", "hidden2.lambda_weight",
                   "hidden2.tau_weight", "hidden2.lambda_bias", "hidden2.tau_bias", "out.lambda_weight", "out.tau_weight", "out.lambda_bias", "out.tau_bias"]
layers = ["hidden1", "hidden2", "out"]
types = ["weight", "bias"]
pyro.clear_param_store()

"""
INITIAL PARAMETRES
"""
init_hidden1_lambdas_weight = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample(torch.Size([L, 28*28]))
init_hidden1_lambdas_bias = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample(torch.Size([L]))
init_hidden2_lambdas_weight = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample(torch.Size([L, L]))
init_hidden2_lambdas_bias = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample(torch.Size([L]))
init_out_lambdas_weight = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample(torch.Size([10,L]))
init_out_lambdas_bias = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample(torch.Size([10]))

init_hidden1_tau_weight = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample()
init_hidden1_tau_bias = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample()
init_hidden2_tau_weight = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample()
init_hidden2_tau_bias = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample()
init_out_tau_weight = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample()
init_out_tau_bias = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample()

init_hidden1_weight = torch.distributions.normal.Normal(0,1).sample(torch.tensor([L,28*28]))
init_hidden2_weight = torch.distributions.normal.Normal(0,1).sample(torch.tensor([L,L]))
init_hidden1_bias = torch.distributions.normal.Normal(0,1).sample(torch.tensor([L]))
init_hidden2_bias = torch.distributions.normal.Normal(0,1).sample(torch.tensor([L]))
init_out_weight = torch.distributions.normal.Normal(0,1).sample(torch.tensor([10,L]))
init_out_bias = torch.distributions.normal.Normal(0,1).sample(torch.tensor([10]))


initial_values = {"hidden1.lambda_weight" : init_hidden1_lambdas_weight,
"hidden1.lambda_bias" : init_hidden1_lambdas_bias,
"hidden2.lambda_weight" : init_hidden2_lambdas_weight,
"hidden2.lambda_bias" : init_hidden2_lambdas_bias,
"out.lambda_weight" : init_out_lambdas_weight,
"out.lambda_bias" : init_out_lambdas_bias,
"hidden1.tau_weight" : init_hidden1_tau_weight,
"hidden1.tau_bias" : init_hidden1_tau_bias,
"hidden2.tau_weight" : init_hidden2_tau_weight,
"hidden2.tau_bias" : init_hidden2_tau_bias,
"out.tau_weight" : init_out_tau_weight,
"out.tau_bias" : init_out_tau_bias,
"hidden1.weight" : init_hidden1_weight,
"hidden1.bias" : init_hidden1_bias,
"hidden2.weight" : init_hidden2_weight,
"hidden2.bias" : init_hidden2_bias,
"out.weight" : init_out_weight,
"out.bias" : init_out_bias}

initial_parameters = {"hidden1.weight" : init_hidden1_weight,
"hidden2.weight" : init_hidden2_weight,
"hidden1.bias" : init_hidden1_bias,
"hidden2.bias" : init_hidden2_bias,
"out.weight" : init_out_weight,
"out.bias" : init_out_bias}

initial_hyperparameters = {"hidden1.lambda_weight" : init_hidden1_lambdas_weight,
"hidden1.lambda_bias" : init_hidden1_lambdas_bias,
"hidden2.lambda_weight" : init_hidden2_lambdas_weight,
"hidden2.lambda_bias" : init_hidden2_lambdas_bias,
"out.lambda_weight" : init_out_lambdas_weight,
"out.lambda_bias" : init_out_lambdas_bias,
"hidden1.tau_weight" : init_hidden1_tau_weight,
"hidden1.tau_bias" : init_hidden1_tau_bias,
"hidden2.tau_weight" : init_hidden2_tau_weight,
"hidden2.tau_bias" : init_hidden2_tau_bias,
"out.tau_weight" : init_out_tau_weight,
"out.tau_bias" : init_out_tau_bias}

lista = ["hidden1.weight", "hidden1.bias", "hidden2.weight", "hidden2.bias", "out.weight", "out.bias", "hidden1.lambda_weight", "hidden1.tau_weight",
         "hidden1.lambda_bias", "hidden1.tau_bias","hidden2.lambda_weight", "hidden2.tau_weight",
         "hidden2.lambda_bias", "hidden2.tau_bias", "out.lambda_weight", "out.tau_weight", "out.lambda_bias", "out.tau_bias"]

print("hola")
boom_kernel = Boomerang(bnn, Sigma=np.eye(478410), refresh_rate = 100, gibbs_rate=100,
                        parameter_list=parameters, hyperparameter_list=hyperparameters, list_of_layers=layers,
                        list_of_types_of_parametres=types, RW_scale=1, initial_parameters=initial_values,
                        init_strategy=init_to_value(values=initial_values),batch_size=100)
print("done")
mcmc_boomerang = MCMC(boom_kernel, num_samples=1, warmup_steps=1)
mcmc_boomerang.run(x_train, y_train)