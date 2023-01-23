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
PATH = os.getcwd()

NUM_SAMPLES = 2000
WARM_UP = 1000

x_train = -2*torch.pi*torch.rand(1000) + torch.pi
y_train = torch.sin(x_train) + torch.normal(0., 0.1, size=(1000,))

x_test = -2*torch.pi*torch.rand(100) + torch.pi
y_test = torch.sin(x_test)

x_extended = torch.linspace(torch.pi, 2*torch.pi, 100)
y_extended = torch.sin(x_extended)
from Pyro_Boomerang import Boomerang
from TrajectorySample import TrajectorySample

"BNN"
class BNN(PyroModule):
  def __init__(self):
    super().__init__()

    # Hidden layer
    self.hidden = PyroModule[nn.Linear](1, 50)
    # Weight priors
    self.hidden.lambda_h_w = PyroSample(dist.HalfCauchy(1.).expand([50, 1]).to_event(2))
    self.hidden.tau_h_w = PyroSample(dist.HalfCauchy(1))
    self.hidden.weight = PyroSample(
      lambda hidden: dist.Normal(0., (hidden.lambda_h_w) * (hidden.tau_h_w)).expand([50, 1]).to_event(2))
    # Bias priors
    self.hidden.lambda_h_b = PyroSample(dist.HalfCauchy(1.).expand([50]).to_event(1))
    self.hidden.tau_h_b = PyroSample(dist.HalfCauchy(1))
    self.hidden.bias = PyroSample(
      lambda hidden: dist.Normal(0., (hidden.lambda_h_b) * (hidden.tau_h_b)).expand([50]).to_event(1))

    # Output layer
    self.out = PyroModule[nn.Linear](50, 1)
    # Weight priors
    self.out.lambda_h_w = PyroSample(dist.HalfCauchy(1.).expand([1, 50]).to_event(2))
    self.out.tau_h_w = PyroSample(dist.HalfCauchy(1))
    self.out.weight = PyroSample(
      lambda out: dist.Normal(0., (out.lambda_h_w) * (out.tau_h_w)).expand([1, 50]).to_event(2))
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

bnn = BNN()
pyro.clear_param_store()
boom_kernel = Boomerang(bnn, Sigma=np.eye(306), refresh_rate = 100, ihpp_sampler='Corbella')
mcmc_boomerang = MCMC(boom_kernel, num_samples=NUM_SAMPLES, warmup_steps=WARM_UP)
mcmc_boomerang.run(x_train.reshape(-1,1), y_train)
trajectory = TrajectorySample(boom_kernel, mcmc_boomerang.get_samples(), NUM_SAMPLES)

nuts_kernel = NUTS(bnn)
mcmc_nuts = MCMC(nuts_kernel, num_samples=NUM_SAMPLES, warmup_steps=WARM_UP)
mcmc_nuts.run(x_train.reshape(-1,1), y_train)


# Set model to evaluation mode
bnn.eval()

# Set up predictive distributions
predictive_boomerang = pyro.infer.Predictive(bnn, posterior_samples = trajectory.sample)
predictive_NUTS = pyro.infer.Predictive(bnn, posterior_samples = trajectory.sample)

## Training predictions
y_train_pred_boomerang = torch.mean(predictive_boomerang(x_train.reshape(-1,1))['obs'], dim=0)
y_train_pred_NUTS = torch.mean(predictive_NUTS(x_train.reshape(-1,1))['obs'], dim=0)


pred_train_summary_boomerang = summary(predictive_boomerang(x_train.reshape(-1,1)))
lower_train_boomerang = pred_train_summary_boomerang["obs"]["5%"]
upper_train_boomerang = pred_train_summary_boomerang["obs"]["95%"]

pred_train_summary_NUTS = summary(predictive_NUTS(x_train.reshape(-1,1)))
lower_train_NUTS = pred_train_summary_NUTS["obs"]["5%"]
upper_train_NUTS = pred_train_summary_NUTS["obs"]["95%"]

## Test predictions
y_test_pred_boomerang = torch.mean(predictive_boomerang(x_test.reshape(-1,1))['obs'], dim=0)

pred_test_summary_boomerang = summary(predictive_boomerang(x_test.reshape(-1,1)))
lower_test_boomerang = pred_test_summary_boomerang["obs"]["5%"]
upper_test_boomerang = pred_test_summary_boomerang["obs"]["95%"]


y_test_pred_NUTS = torch.mean(predictive_NUTS(x_test.reshape(-1,1))['obs'], dim=0)

pred_test_summary_NUTS = summary(predictive_NUTS(x_test.reshape(-1,1)))
lower_test_NUTS = pred_test_summary_NUTS["obs"]["5%"]
upper_test_NUTS = pred_test_summary_NUTS["obs"]["95%"]

## Extended predictions
y_ext_pred_boomerang = torch.mean(predictive_boomerang(x_extended.reshape(-1,1))['obs'], dim=0)

pred_ext_summary_boomerang = summary(predictive_boomerang(x_extended.reshape(-1,1)))
lower_ext_boomerang = pred_ext_summary_boomerang["obs"]["5%"]
upper_ext_boomerang = pred_ext_summary_boomerang["obs"]["95%"]

y_ext_pred_NUTS = torch.mean(predictive_NUTS(x_extended.reshape(-1,1))['obs'], dim=0)

pred_ext_summary_NUTS = summary(predictive_NUTS(x_extended.reshape(-1,1)))
lower_ext_NUTS = pred_ext_summary_NUTS["obs"]["5%"]
upper_ext_NUTS = pred_ext_summary_NUTS["obs"]["95%"]


#saving data
#TRAIN



with open(PATH + "/results/boomerang/postSample_boomerang.csv", 'w') as f:
    for key in trajectory.sample.keys():
        f.write("%s,%s\n"%(key,trajectory.sample[key]))

with open(PATH + "/results/NUTS/postSample_NUTS.csv", 'w') as f:
  for key in mcmc_nuts.get_samples().keys():
    f.write("%s,%s\n" % (key, mcmc_nuts.get_samples()[key]))

x_train_df = pd.DataFrame(x_train)
x_train_df.to_csv(PATH + "/results/x_train.csv")

y_train_df = pd.DataFrame(y_train)
y_train_df.to_csv(PATH + "/results/y_train.csv")

y_train_pred_df_boomerang = pd.DataFrame(y_train_pred_boomerang)
y_train_pred_df_boomerang.to_csv(PATH + "/results/boomerang/y_train_pred_boomerang.csv")

y_train_pred_df_NUTS = pd.DataFrame(y_train_pred_NUTS)
y_train_pred_df_NUTS.to_csv(PATH + "/results/nuts/y_train_pred_NUTS.csv")

lower_train_df_boomerang = pd.DataFrame(lower_train_boomerang)
lower_train_df_boomerang.to_csv(PATH + "/results/boomerang/lower_train_boomerang.csv")

upper_train_df_boomerang = pd.DataFrame(upper_train_boomerang)
upper_train_df_boomerang.to_csv(PATH + "/results/boomerang/upper_train_boomerang.csv")

lower_train_df_NUTS = pd.DataFrame(lower_train_NUTS)
lower_train_df_NUTS.to_csv(PATH + "/results/NUTS/lower_train_NUTS.csv")

upper_train_df_NUTS = pd.DataFrame(upper_train_NUTS)
upper_train_df_NUTS.to_csv(PATH + "/results/NUTS/upper_train_NUTS.csv")



#TEST
x_test_df = pd.DataFrame(x_test)
x_test_df.to_csv(PATH + "/results/x_test.csv")

y_test_df = pd.DataFrame(y_test)
y_test_df.to_csv(PATH + "/results/y_test.csv")

y_test_pred_df_boomerang = pd.DataFrame(y_test_pred_boomerang)
y_test_pred_df_boomerang.to_csv(PATH + "/results/boomerang/y_test_pred_boomerang.csv")

y_test_pred_df_NUTS = pd.DataFrame(y_test_pred_NUTS)
y_test_pred_df_NUTS.to_csv(PATH + "/results/nuts/y_test_pred_NUTS.csv")

lower_test_df_boomerang = pd.DataFrame(lower_test_boomerang)
lower_test_df_boomerang.to_csv(PATH + "/results/boomerang/lower_test_boomerang.csv")

upper_test_df_boomerang = pd.DataFrame(upper_test_boomerang)
upper_test_df_boomerang.to_csv(PATH + "/results/boomerang/upper_test_boomerang.csv")

lower_test_df_NUTS = pd.DataFrame(lower_test_NUTS)
lower_test_df_NUTS.to_csv(PATH + "/results/NUTS/lower_test_NUTS.csv")

upper_test_df_NUTS = pd.DataFrame(upper_test_NUTS)
upper_test_df_NUTS.to_csv(PATH + "/results/NUTS/upper_test_NUTS.csv")



#EXTENDED
x_ext_df = pd.DataFrame(x_extended)
x_ext_df.to_csv(PATH + "/results/x_ext.csv")

y_ext_df = pd.DataFrame(y_extended)
y_ext_df.to_csv(PATH + "/results/y_ext.csv")

y_ext_pred_df_boomerang = pd.DataFrame(y_ext_pred_boomerang)
y_ext_pred_df_boomerang.to_csv(PATH + "/results/boomerang/y_ext_pred_boomerang.csv")

y_ext_pred_df_NUTS = pd.DataFrame(y_ext_pred_NUTS)
y_ext_pred_df_NUTS.to_csv(PATH + "/results/nuts/y_ext_pred_NUTS.csv")

lower_ext_df_boomerang = pd.DataFrame(lower_ext_boomerang)
lower_ext_df_boomerang.to_csv(PATH + "/results/boomerang/lower_ext_boomerang.csv")

upper_ext_df_boomerang = pd.DataFrame(upper_ext_boomerang)
upper_ext_df_boomerang.to_csv(PATH + "/results/boomerang/upper_ext_boomerang.csv")

lower_ext_df_NUTS = pd.DataFrame(lower_ext_NUTS)
lower_ext_df_NUTS.to_csv(PATH + "/results/NUTS/lower_ext_NUTS.csv")

upper_ext_df_NUTS = pd.DataFrame(upper_ext_NUTS)
upper_ext_df_NUTS.to_csv(PATH + "/results/NUTS/upper_ext_NUTS.csv")




