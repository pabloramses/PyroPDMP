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
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer import MCMC, NUTS
PATH = os.getcwd()

x = torch.linspace(-np.pi, np.pi, 100)
y = torch.sin(x) + torch.normal(0., 0.1, size=(100,))

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
mcmc = MCMC(boom_kernel, num_samples=10, warmup_steps=2)
mcmc.run(x.reshape(-1,1), y)

# Set model to evaluation mode
bnn.eval()

# Set up predictive distribution
predictive = pyro.infer.Predictive(bnn, posterior_samples = mcmc.get_samples())
## Predictions
y_train_pred = torch.mean(predictive(x.reshape(-1,1))['obs'], dim=0)


""""
def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats

pred_summary = summary(predictive(torch.tensor(x).detach().float().reshape(-1,1)))
lower = pred_summary["obs"]["5%"]
upper = pred_summary["obs"]["95%"]
"""
y_train_pred_df = pd.DataFrame(y_train_pred)
y_train_pred_df.to_csv(PATH + "/y_train_pred.csv")

"""
lower_df = pd.DataFrame(lower)
lower_df.to_csv(PATH + "/lower.csv")
upper_df = pd.DataFrame(upper)
upper_df.to_csv(PATH + "/upper.csv")
"""