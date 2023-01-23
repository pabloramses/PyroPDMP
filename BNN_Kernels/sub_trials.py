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

NUM_SAMPLES = 1000
WARM_UP = 1000
batch_size = 100
x_train = -2*torch.pi*torch.rand(100) + torch.pi
y_train = torch.sin(x_train) + torch.normal(0., 0.1, size=(100,))

from Boomerang import Boomerang
#from Pyro_Boomerang import Boomerang
from TrajectorySample import TrajectorySample


L=100
"BNN"
class BNN(PyroModule):
  def __init__(self):
    super().__init__()

    # Hidden layer
    self.hidden = PyroModule[nn.Linear](1, L)
    # Weight priors
    self.hidden.lambda_weight = PyroSample(dist.HalfCauchy(1.).expand([L, 1]).to_event(2))
    self.hidden.tau_weight = PyroSample(dist.HalfCauchy(1))
    self.hidden.weight = PyroSample(
      lambda hidden: dist.Normal(0., (hidden.lambda_weight) * (hidden.tau_weight)).expand([L, 1]).to_event(2))
    # Bias priors
    self.hidden.lambda_bias = PyroSample(dist.HalfCauchy(1.).expand([L]).to_event(1))
    self.hidden.tau_bias = PyroSample(dist.HalfCauchy(1))
    self.hidden.bias = PyroSample(
      lambda hidden: dist.Normal(0., (hidden.lambda_bias) * (hidden.tau_bias)).expand([L]).to_event(1))

    # Output layer
    self.out = PyroModule[nn.Linear](L, 1)
    # Weight priors
    self.out.lambda_weight = PyroSample(dist.HalfCauchy(1.).expand([1, L]).to_event(2))
    self.out.tau_weight = PyroSample(dist.HalfCauchy(1))
    self.out.weight = PyroSample(
      lambda out: dist.Normal(0., (out.lambda_weight) * (out.tau_weight)).expand([1, L]).to_event(2))
    # Bias priors
    self.out.lambda_bias = PyroSample(dist.HalfCauchy(1.).expand([1]).to_event(1))
    self.out.tau_bias = PyroSample(dist.HalfCauchy(1))
    self.out.bias = PyroSample(lambda out: dist.Normal(0., (out.lambda_bias) * (out.tau_bias)).expand([1]).to_event(1))

  def forward(self, x, y=None):
    x = F.relu(self.hidden(x))
    mu = self.out(x).squeeze()
    # Likelihood
    with pyro.plate("data", x.shape[0]):
      obs = pyro.sample("obs", dist.Normal(mu, 0.1), obs=y)
    return mu

bnn = BNN()

parameters = ["hidden.weight", "hidden.bias", "out.weight", "out.bias"]
hyperparameters = ["hidden.lambda_weight", "hidden.tau_weight", "hidden.lambda_bias", "hidden.tau_bias", "out.lambda_weight", "out.tau_weight", "out.lambda_bias", "out.tau_bias"]
layers = ["hidden", "out"]
types = ["weight", "bias"]
pyro.clear_param_store()

"""
INITIAL PARAMETRES
"""
init_hidden_lambdas_weight = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample(torch.Size([L, 1]))
init_hidden_lambdas_bias = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample(torch.Size([L]))
init_out_lambdas_weight = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample(torch.Size([1,L]))
init_out_lambdas_bias = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample(torch.Size([1]))

init_hidden_tau_weight = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample()
init_hidden_tau_bias = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample()
init_out_tau_weight = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample()
init_out_tau_bias = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample()

init_hidden_weight = torch.distributions.normal.Normal(0,1).sample(torch.tensor([L,1]))
init_hidden_bias = torch.distributions.normal.Normal(0,1).sample(torch.tensor([L]))
init_out_weight = torch.distributions.normal.Normal(0,1).sample(torch.tensor([1,L]))
init_out_bias = torch.distributions.normal.Normal(0,1).sample(torch.tensor([1]))


initial_values = {"hidden.lambda_weight" : init_hidden_lambdas_weight,
"hidden.lambda_bias" : init_hidden_lambdas_bias,
"out.lambda_weight" : init_out_lambdas_weight,
"out.lambda_bias" : init_out_lambdas_bias,
"hidden.tau_weight" : init_hidden_tau_weight,
"hidden.tau_bias" : init_hidden_tau_bias,
"out.tau_weight" : init_out_tau_weight,
"out.tau_bias" : init_out_tau_bias,
"hidden.weight" : init_hidden_weight,
"hidden.bias" : init_hidden_bias,
"out.weight" : init_out_weight,
"out.bias" : init_out_bias}

initial_parameters = {"hidden.weight" : init_hidden_weight,
"hidden.bias" : init_hidden_bias,
"out.weight" : init_out_weight,
"out.bias" : init_out_bias}

initial_hyperparameters = {"hidden.lambda_weight" : init_hidden_lambdas_weight,
"hidden.lambda_bias" : init_hidden_lambdas_bias,
"out.lambda_weight" : init_out_lambdas_weight,
"out.lambda_bias" : init_out_lambdas_bias,
"hidden.tau_weight" : init_hidden_tau_weight,
"hidden.tau_bias" : init_hidden_tau_bias,
"out.tau_weight" : init_out_tau_weight,
"out.tau_bias" : init_out_tau_bias}

lista = ["hidden.weight", "hidden.bias", "out.weight", "out.bias", "hidden.lambda_weight", "hidden.tau_weight", "hidden.lambda_bias",
              "hidden.tau_bias", "out.lambda_weight", "out.tau_weight", "out.lambda_bias", "out.tau_bias"]

boom_kernel = Boomerang(bnn, Sigma=np.eye(3*L+1), refresh_rate = 100, gibbs_rate=100,
                        parameter_list=parameters, hyperparameter_list=hyperparameters, list_of_layers=layers,
                        list_of_types_of_parametres=types, RW_scale=1, initial_parameters=initial_values,
                        init_strategy=init_to_value(values=initial_values),batch_size=100)

t0 = time.time()
mcmc_boomerang = MCMC(boom_kernel, num_samples=NUM_SAMPLES, warmup_steps=WARM_UP)
mcmc_boomerang.run(x_train.reshape(-1,1), y_train)
print("clock time", time.time()-t0)


"""
t0 = time.time()
nuts_kernel = NUTS(bnn)
mcmc_nuts = MCMC(nuts_kernel, num_samples=75, warmup_steps=75)
mcmc_nuts.run(x_train.reshape(-1,1), y_train)
print("clock time", time.time()-t0)
"""

"""
full_boomerang = Boomerang_2(bnn, Sigma=np.eye(6*L+6), refresh_rate = 100, ihpp_sampler='Corbella')
mcmc_boomerang = MCMC(full_boomerang, num_samples=NUM_SAMPLES, warmup_steps=WARM_UP)
mcmc_boomerang.run(x_train.reshape(-1,1), y_train)
trajectory = TrajectorySample(full_boomerang, mcmc_boomerang.get_samples(), NUM_SAMPLES)
"""


"""
v =  np.random.normal(0, 1, 3*L+1)

v_2 =  np.random.normal(0, 1, 18)

ts = np.linspace(0,4,100)
rates_gibbs= []
rates_nongibbs = []
for i in range(len(ts)):
  rates_gibbs.append(boom_kernel.rate_of_t_gibbs(boom_kernel.dict_of_tensors_to_numpy(initial_parameters),initial_hyperparameters, v, ts[i]))
  rates_nongibbs.append(
    boom_kernel.rate_of_t_nongibbs(boom_kernel.dict_of_tensors_to_numpy(initial_values), v_2, lista,
                                ts[i]))

plt.plot(rates_gibbs)
plt.plot(rates_nongibbs)
plt.show()
#rate = boom_kernel.rate_of_t_gibbs(boom_kernel.dict_of_tensors_to_numpy(initial_parameters),initial_hyperparameters, v, 0.3)
#print(rate)
"""



## Integrate over hyperparametres
parameter_sample = {}
for key in parameters:
  parameter_sample.update({key: mcmc_boomerang.get_samples()[key]})

trajectory = TrajectorySample(boom_kernel, parameter_sample, NUM_SAMPLES)
predictive_boomerang = pyro.infer.Predictive(bnn, posterior_samples = trajectory.sample)

## Training predictions
y_train_pred_boomerang = torch.mean(predictive_boomerang(x_train.reshape(-1,1))['obs'], dim=0)

pred_train_summary_boomerang = summary(predictive_boomerang(x_train.reshape(-1,1)))
lower_train_boomerang = pred_train_summary_boomerang["obs"]["5%"]
upper_train_boomerang = pred_train_summary_boomerang["obs"]["95%"]





#saving data
#TRAIN



with open(PATH + "/results/boomerang/postSample_boomerang.csv", 'w') as f:
    for key in trajectory.sample.keys():
        f.write("%s,%s\#n"%(key,trajectory.sample[key])) #QUITAR LA ALMOHADILLA

x_train_df = pd.DataFrame(x_train)
x_train_df.to_csv(PATH + "/results/x_train.csv")

y_train_df = pd.DataFrame(y_train)
y_train_df.to_csv(PATH + "/results/y_train.csv")

y_train_pred_df_boomerang = pd.DataFrame(y_train_pred_boomerang)
y_train_pred_df_boomerang.to_csv(PATH + "/results/boomerang/y_train_pred_boomerang.csv")

lower_train_df_boomerang = pd.DataFrame(lower_train_boomerang)
lower_train_df_boomerang.to_csv(PATH + "/results/boomerang/lower_train_boomerang.csv")

upper_train_df_boomerang = pd.DataFrame(upper_train_boomerang)
upper_train_df_boomerang.to_csv(PATH + "/results/boomerang/upper_train_boomerang.csv")

"""
# Set model to evaluation mode
bnn.eval()

# Set up predictive distributions
predictive_NUTS = pyro.infer.Predictive(bnn, posterior_samples = mcmc_nuts.get_samples())

## Training predictions
y_train_pred_NUTS = torch.mean(predictive_NUTS(x_train.reshape(-1,1))['obs'], dim=0)


pred_train_summary_NUTS = summary(predictive_NUTS(x_train.reshape(-1,1)))
lower_train_NUTS = pred_train_summary_NUTS["obs"]["5%"]
upper_train_NUTS = pred_train_summary_NUTS["obs"]["95%"]

with open(PATH + "/results/NUTS/postSample_NUTS.csv", 'w') as f:
  for key in mcmc_nuts.get_samples().keys():
    f.write("%s,%s\n" % (key, mcmc_nuts.get_samples()[key]))

x_train_df = pd.DataFrame(x_train)
x_train_df.to_csv(PATH + "/results/x_train.csv")

y_train_df = pd.DataFrame(y_train)
y_train_df.to_csv(PATH + "/results/y_train.csv")

y_train_pred_df_NUTS = pd.DataFrame(y_train_pred_NUTS)
y_train_pred_df_NUTS.to_csv(PATH + "/results/nuts/y_train_pred_NUTS.csv")


lower_train_df_NUTS = pd.DataFrame(lower_train_NUTS)
lower_train_df_NUTS.to_csv(PATH + "/results/NUTS/lower_train_NUTS.csv")

upper_train_df_NUTS = pd.DataFrame(upper_train_NUTS)
upper_train_df_NUTS.to_csv(PATH + "/results/NUTS/upper_train_NUTS.csv")
"""
