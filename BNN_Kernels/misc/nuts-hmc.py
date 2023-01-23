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
from collections import OrderedDict
from pyro.ops import stats
import warnings
import traceback as tb


def _safe(fn):
  """
  Safe version of utilities in the :mod:`pyro.ops.stats` module. Wrapped
  functions return `NaN` tensors instead of throwing exceptions.

  :param fn: stats function from :mod:`pyro.ops.stats` module.
  """

  def wrapped(sample, *args, **kwargs):
    try:
      val = fn(sample, *args, **kwargs)
    except Exception:
      warnings.warn(tb.format_exc())
      val = torch.full(
        sample.shape[2:], float("nan"), dtype=sample.dtype, device=sample.device
      )
    return val

  return wrapped
def diagnostics(samples, group_by_chain=True):
  """
  Gets diagnostics statistics such as effective sample size and
  split Gelman-Rubin using the samples drawn from the posterior
  distribution.

  :param dict samples: dictionary of samples keyed by site name.
  :param bool group_by_chain: If True, each variable in `samples`
      will be treated as having shape `num_chains x num_samples x sample_shape`.
      Otherwise, the corresponding shape will be `num_samples x sample_shape`
      (i.e. without chain dimension).
  :return: dictionary of diagnostic stats for each sample site.
  """
  diagnostics = {}
  for site, support in samples.items():
    if not group_by_chain:
      support = support.unsqueeze(0)
    site_stats = OrderedDict()
    site_stats["n_eff"] = _safe(stats.effective_sample_size)(support)
    diagnostics[site] = site_stats
  return diagnostics


def diagnostics_class(mcmc):
  """
  Gets some diagnostics statistics such as effective sample size, split
  Gelman-Rubin, or divergent transitions from the sampler.
  """
  diag = diagnostics(mcmc._samples)
  for diag_name in mcmc._diagnostics[0]:
    diag[diag_name] = {
      "chain {}".format(i): mcmc._diagnostics[i][diag_name]
      for i in range(mcmc.num_chains)
    }
  return diag


print("NUTS START")
torch.manual_seed(0)
x_train = -2*torch.pi*torch.rand(1000) + torch.pi
torch.manual_seed(0)
y_train = torch.sin(x_train) + torch.normal(0., 0.1, size=(1000,))

torch.manual_seed(1)
x_test = -2*torch.pi*torch.rand(100) + torch.pi
torch.manual_seed(1)
y_test = torch.sin(x_test)


NUM_SAMPLES = 2000
WARM_UP = 1000

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

L=10
bnn_10 = BNN(L)
pyro.clear_param_store()

nuts_kernel_10 = NUTS(bnn_10)
mcmc_NUTS_10 = MCMC(nuts_kernel_10, num_samples=NUM_SAMPLES, warmup_steps=WARM_UP)
mcmc_NUTS_10.run(x_train.reshape(-1,1), y_train)

for k in list(mcmc_NUTS_10.get_samples().keys()):
  print(k)
  print(mcmc_NUTS_10.get_samples()[k].mean(0))

print(diagnostics_class(mcmc_NUTS_10))

predictive_NUTS_10 = pyro.infer.Predictive(bnn_10, posterior_samples = mcmc_NUTS_10.get_samples())
## Test predictions
y_test_pred_NUTS_10 = torch.mean(predictive_NUTS_10(x_test.reshape(-1,1))['obs'], dim=0)

MSE_test_10 = torch.mean((y_test_pred_NUTS_10-y_test)**2)

pred_test_summary_NUTS_10 = summary(predictive_NUTS_10(x_test.reshape(-1,1)))
lower_test_NUTS_10 = pred_test_summary_NUTS_10["obs"]["5%"]
upper_test_NUTS_10 = pred_test_summary_NUTS_10["obs"]["95%"]

perCorrect_10 = percentage_correct(lower_test_NUTS_10, upper_test_NUTS_10, y_test)

print("The MSE for 10 units was ", MSE_test_10, " and the percentage correct ", perCorrect_10)


L=50
bnn_50 = BNN(L)
pyro.clear_param_store()

nuts_kernel_50 = NUTS(bnn_50)
mcmc_NUTS_50 = MCMC(nuts_kernel_50, num_samples=NUM_SAMPLES, warmup_steps=WARM_UP)
mcmc_NUTS_50.run(x_train.reshape(-1,1), y_train)

for k in list(mcmc_NUTS_50.get_samples().keys()):
  print(k)
  print(mcmc_NUTS_50.get_samples()[k].mean(0))

print(diagnostics_class(mcmc_NUTS_50))
predictive_NUTS_50 = pyro.infer.Predictive(bnn_50, posterior_samples = mcmc_NUTS_50.get_samples())
## Test predictions
y_test_pred_NUTS_50 = torch.mean(predictive_NUTS_50(x_test.reshape(-1,1))['obs'], dim=0)

MSE_test_50 = torch.mean((y_test_pred_NUTS_50-y_test)**2)

pred_test_summary_NUTS_50 = summary(predictive_NUTS_50(x_test.reshape(-1,1)))
lower_test_NUTS_50 = pred_test_summary_NUTS_50["obs"]["5%"]
upper_test_NUTS_50 = pred_test_summary_NUTS_50["obs"]["95%"]

perCorrect_50 = percentage_correct(lower_test_NUTS_50, upper_test_NUTS_50, y_test)

print("The MSE for 50 units was ", MSE_test_50, " and the percentage correct ", perCorrect_50)

L=100
bnn_100 = BNN(L)
pyro.clear_param_store()

nuts_kernel_100 = NUTS(bnn_100)
mcmc_NUTS_100 = MCMC(nuts_kernel_100, num_samples=NUM_SAMPLES, warmup_steps=WARM_UP)
mcmc_NUTS_100.run(x_train.reshape(-1,1), y_train)

for k in list(mcmc_NUTS_100.get_samples().keys()):
  print(k)
  print(mcmc_NUTS_100.get_samples()[k].mean(0))

print(diagnostics_class(mcmc_NUTS_100))
predictive_NUTS_100 = pyro.infer.Predictive(bnn_100, posterior_samples = mcmc_NUTS_100.get_samples())
## Test predictions
y_test_pred_NUTS_100 = torch.mean(predictive_NUTS_100(x_test.reshape(-1,1))['obs'], dim=0)

MSE_test_100 = torch.mean((y_test_pred_NUTS_100-y_test)**2)

pred_test_summary_NUTS_100 = summary(predictive_NUTS_100(x_test.reshape(-1,1)))
lower_test_NUTS_100 = pred_test_summary_NUTS_100["obs"]["5%"]
upper_test_NUTS_100 = pred_test_summary_NUTS_100["obs"]["95%"]

perCorrect_100 = percentage_correct(lower_test_NUTS_100, upper_test_NUTS_100, y_test)

print("The MSE for 100 units was ", MSE_test_100, " and the percentage correct ", perCorrect_100)

print("END NUTS")