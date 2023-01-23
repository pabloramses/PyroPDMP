import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import MCMC, NUTS
from utils import *
import os
from Pyro_Boomerang import Boomerang
from TrajectorySample import TrajectorySample

print("BOOMERANG START")
torch.manual_seed(0)
x_train = -2*torch.pi*torch.rand(1000) + torch.pi
torch.manual_seed(0)
y_train = torch.sin(x_train) + torch.normal(0., 0.1, size=(1000,))

torch.manual_seed(1)
x_test = -2*torch.pi*torch.rand(100) + torch.pi
torch.manual_seed(1)
y_test = torch.sin(x_test)


NUM_SAMPLES = 2000
WARM_UP = 2000

class BNN(PyroModule):
  def __init__(self, L):
    super().__init__()

    # Hidden layer
    self.hidden = PyroModule[nn.Linear](1, L)
    # Weight priors
    self.hidden.lambda_h_w = PyroSample(dist.HalfCauchy(1.).expand([L, 1]).to_event(2))
    self.hidden.tau_h_w = PyroSample(dist.HalfCauchy(1))
    self.hidden.weight = PyroSample(
      lambda hidden: dist.Normal(0., (hidden.lambda_h_w) * (hidden.tau_h_w)).expand([L, 1]).to_event(2))
    # Bias priors
    self.hidden.lambda_h_b = PyroSample(dist.HalfCauchy(1.).expand([L]).to_event(1))
    self.hidden.tau_h_b = PyroSample(dist.HalfCauchy(1))
    self.hidden.bias = PyroSample(
      lambda hidden: dist.Normal(0., (hidden.lambda_h_b) * (hidden.tau_h_b)).expand([L]).to_event(1))

    # Output layer
    self.out = PyroModule[nn.Linear](L, 1)
    # Weight priors
    self.out.lambda_h_w = PyroSample(dist.HalfCauchy(1.).expand([1, L]).to_event(2))
    self.out.tau_h_w = PyroSample(dist.HalfCauchy(1))
    self.out.weight = PyroSample(
      lambda out: dist.Normal(0., (out.lambda_h_w) * (out.tau_h_w)).expand([1, L]).to_event(2))
    # Bias priors
    self.out.lambda_h_b = PyroSample(dist.HalfCauchy(1.).expand([1]).to_event(1))
    self.out.tau_h_b = PyroSample(dist.HalfCauchy(1))
    self.out.bias = PyroSample(lambda out: dist.Normal(0., (out.lambda_h_b) * (out.tau_h_b)).expand([1]).to_event(1))

  def forward(self, x, y=None):
    x = F.relu(self.hidden(x))
    mu = self.out(x).squeeze()
    # Likelihood
    with pyro.plate("data", x.shape[0]):
      obs = pyro.sample("obs", dist.Normal(mu, 0.1), obs=y)
    return mu

"""
L=10
bnn_10 = BNN(L)
pyro.clear_param_store()

boom_kernel_10 = Boomerang(bnn_10, Sigma=np.eye(6*L+6), refresh_rate = 50, ihpp_sampler='Corbella')
mcmc_boomerang_10 = MCMC(boom_kernel_10, num_samples=NUM_SAMPLES, warmup_steps=WARM_UP)
mcmc_boomerang_10.run(x_train.reshape(-1,1), y_train)
trajectory_10 = TrajectorySample(boom_kernel_10, mcmc_boomerang_10.get_samples(), NUM_SAMPLES)

predictive_boomerang_10 = pyro.infer.Predictive(bnn_10, posterior_samples = trajectory_10.sample)
## Test predictions
y_test_pred_boomerang_10 = torch.mean(predictive_boomerang_10(x_test.reshape(-1,1))['obs'], dim=0)

MSE_test_10 = torch.mean((y_test_pred_boomerang_10-y_test)**2)

pred_test_summary_boomerang_10 = summary(predictive_boomerang_10(x_test.reshape(-1,1)))
lower_test_boomerang_10 = pred_test_summary_boomerang_10["obs"]["5%"]
upper_test_boomerang_10 = pred_test_summary_boomerang_10["obs"]["95%"]

perCorrect_10 = percentage_correct(lower_test_boomerang_10, upper_test_boomerang_10, y_test)

print("The MSE for 10 units was ", MSE_test_10, " and the percentage correct ", perCorrect_10)


L=50
bnn_50 = BNN(L)
pyro.clear_param_store()

boom_kernel_50 = Boomerang(bnn_50, Sigma=np.eye(6*L+6), refresh_rate = 75, ihpp_sampler='Corbella')
mcmc_boomerang_50 = MCMC(boom_kernel_50, num_samples=NUM_SAMPLES, warmup_steps=WARM_UP)
mcmc_boomerang_50.run(x_train.reshape(-1,1), y_train)
trajectory_50 = TrajectorySample(boom_kernel_50, mcmc_boomerang_50.get_samples(), NUM_SAMPLES)

predictive_boomerang_50 = pyro.infer.Predictive(bnn_50, posterior_samples = trajectory_50.sample)
## Test predictions
y_test_pred_boomerang_50 = torch.mean(predictive_boomerang_50(x_test.reshape(-1,1))['obs'], dim=0)

MSE_test_50 = torch.mean((y_test_pred_boomerang_50-y_test)**2)

pred_test_summary_boomerang_50 = summary(predictive_boomerang_50(x_test.reshape(-1,1)))
lower_test_boomerang_50 = pred_test_summary_boomerang_50["obs"]["5%"]
upper_test_boomerang_50 = pred_test_summary_boomerang_50["obs"]["95%"]

perCorrect_50 = percentage_correct(lower_test_boomerang_50, upper_test_boomerang_50, y_test)

print("The MSE for 50 units was ", MSE_test_50, " and the percentage correct ", perCorrect_50)
"""
L=100
bnn_100 = BNN(L)
pyro.clear_param_store()

boom_kernel_100 = Boomerang(bnn_100, Sigma=np.eye(6*L+6), refresh_rate = 200, ihpp_sampler='Corbella')
mcmc_boomerang_100 = MCMC(boom_kernel_100, num_samples=NUM_SAMPLES, warmup_steps=WARM_UP)
mcmc_boomerang_100.run(x_train.reshape(-1,1), y_train)
trajectory_100 = TrajectorySample(boom_kernel_100, mcmc_boomerang_100.get_samples(), NUM_SAMPLES)

predictive_boomerang_100 = pyro.infer.Predictive(bnn_100, posterior_samples = trajectory_100.sample)
## Test predictions
y_test_pred_boomerang_100 = torch.mean(predictive_boomerang_100(x_test.reshape(-1,1))['obs'], dim=0)

MSE_test_100 = torch.mean((y_test_pred_boomerang_100-y_test)**2)

pred_test_summary_boomerang_100 = summary(predictive_boomerang_100(x_test.reshape(-1,1)))
lower_test_boomerang_100 = pred_test_summary_boomerang_100["obs"]["5%"]
upper_test_boomerang_100 = pred_test_summary_boomerang_100["obs"]["95%"]

perCorrect_100 = percentage_correct(lower_test_boomerang_100, upper_test_boomerang_100, y_test)

print("The MSE for 100 units was ", MSE_test_100, " and the percentage correct ", perCorrect_100)

print("END BOOMERANG")