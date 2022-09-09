import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from utils import *
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer import MCMC, NUTS
PATH = os.getcwd()

x_train = torch.linspace(-torch.pi, torch.pi, 100)
y_train = torch.sin(x_train) + torch.normal(0., 0.1, size=(100,))

x_test = -2*torch.pi*torch.rand(100) + torch.pi
y_test = torch.sin(x_test)

x_extended = torch.linspace(torch.pi, 2*torch.pi, 100)
y_extended = torch.sin(x_extended)
from Pyro_Boomerang import Boomerang
"BNN"

class BNN(PyroModule):
  def __init__(self):
    super().__init__()
    self.hidden = PyroModule[nn.Linear](1,100)
    self.hidden.weight = PyroSample(dist.Normal(0., 1.).expand([100,1]).to_event(2))
    self.hidden.bias = PyroSample(dist.Normal(0., 1.).expand([100]).to_event(1))

    self.out = PyroModule[nn.Linear](100,1)
    self.out.weight = PyroSample(dist.Normal(0., 1.).expand([1,100]).to_event(2))
    self.out.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

  def forward(self, x, y=None):
    #x =  F.relu(self.hidden(x))
    x = F.relu(self.hidden(x))
    mu = self.out(x).squeeze()
    #likelihood
    with pyro.plate("data", x.shape[0]):
      obs = pyro.sample("obs", dist.Normal(mu, 0.1), obs=y)
    return mu


bnn = BNN()
pyro.clear_param_store()

boom_kernel = Boomerang(bnn, Sigma=np.eye(301), refresh_rate = 100, ihpp_sampler='Corbella')
from pyro.infer import MCMC
mcmc = MCMC(boom_kernel, num_samples=10000, warmup_steps=2000)
mcmc.run(x_train.reshape(-1,1), y_train)

# Set model to evaluation mode
bnn.eval()

# Set up predictive distribution
predictive = pyro.infer.Predictive(bnn, posterior_samples = mcmc.get_samples())
## Training predictions
y_train_pred = torch.mean(predictive(x_train.reshape(-1,1))['obs'], dim=0)


pred_train_summary = summary(predictive(x_train.reshape(-1,1)))
lower_train = pred_train_summary["obs"]["5%"]
upper_train = pred_train_summary["obs"]["95%"]


variances_train = variance_limits(predictive(x_train.reshape(-1,1)))
var_train_low = variances_train["obs"]["low"]
var_train_up = variances_train["obs"]["up"]

## Test predictions
y_test_pred = torch.mean(predictive(x_test.reshape(-1,1))['obs'], dim=0)

pred_test_summary = summary(predictive(x_test.reshape(-1,1)))
lower_test = pred_test_summary["obs"]["5%"]
upper_test = pred_test_summary["obs"]["95%"]


variances_test = variance_limits(predictive(x_test.reshape(-1,1)))
var_test_low = variances_test["obs"]["low"]
var_test_up = variances_test["obs"]["up"]

## Extended predictions
y_ext_pred = torch.mean(predictive(x_extended.reshape(-1,1))['obs'], dim=0)

pred_ext_summary = summary(predictive(x_extended.reshape(-1,1)))
lower_ext = pred_ext_summary["obs"]["5%"]
upper_ext = pred_ext_summary["obs"]["95%"]


variances_ext = variance_limits(predictive(x_extended.reshape(-1,1)))
var_ext_low = variances_ext["obs"]["low"]
var_ext_up = variances_ext["obs"]["up"]


#saving data
#TRAIN
x_train_df = pd.DataFrame(x_train)
x_train_df.to_csv(PATH + "/results/x_train.csv")

y_train_df = pd.DataFrame(y_train)
y_train_df.to_csv(PATH + "/results/y_train.csv")

y_train_pred_df = pd.DataFrame(y_train_pred)
y_train_pred_df.to_csv(PATH + "/results/y_train_pred.csv")

lower_train_df = pd.DataFrame(lower_train)
lower_train_df.to_csv(PATH + "/results/lower_train.csv")

upper_train_df = pd.DataFrame(upper_train)
upper_train_df.to_csv(PATH + "/results/upper_train.csv")

var_train_low_df = pd.DataFrame(var_train_low)
var_train_low_df.to_csv(PATH + "/results/var_train_low.csv")

var_train_up_df = pd.DataFrame(var_train_up)
var_train_up_df.to_csv(PATH + "/results/var_train_up.csv")

#TEST
x_test_df = pd.DataFrame(x_test)
x_test_df.to_csv(PATH + "/results/x_test.csv")

y_test_df = pd.DataFrame(y_test)
y_test_df.to_csv(PATH + "/results/y_test.csv")

y_test_pred_df = pd.DataFrame(y_test_pred)
y_test_pred_df.to_csv(PATH + "/results/y_test_pred.csv")

lower_test_df = pd.DataFrame(lower_test)
lower_test_df.to_csv(PATH + "/results/lower_test.csv")

upper_test_df = pd.DataFrame(upper_test)
upper_test_df.to_csv(PATH + "/results/upper_test.csv")

var_test_low_df = pd.DataFrame(var_test_low)
var_test_low_df.to_csv(PATH + "/results/var_test_low.csv")

var_test_up_df = pd.DataFrame(var_test_up)
var_test_up_df.to_csv(PATH + "/results/var_test_up.csv")

#EXTENDED
x_ext_df = pd.DataFrame(x_extended)
x_ext_df.to_csv(PATH + "/results/x_ext.csv")

y_ext_df = pd.DataFrame(y_extended)
y_ext_df.to_csv(PATH + "/results/y_ext.csv")

y_ext_pred_df = pd.DataFrame(y_ext_pred)
y_ext_pred_df.to_csv(PATH + "/results/y_ext_pred.csv")

lower_ext_df = pd.DataFrame(lower_ext)
lower_ext_df.to_csv(PATH + "/results/lower_ext.csv")

upper_ext_df = pd.DataFrame(upper_ext)
upper_ext_df.to_csv(PATH + "/results/upper_ext.csv")

var_ext_low_df = pd.DataFrame(var_ext_low)
var_ext_low_df.to_csv(PATH + "/results/var_ext_low.csv")

var_ext_up_df = pd.DataFrame(var_ext_up)
var_ext_up_df.to_csv(PATH + "/results/var_ext_up.csv")