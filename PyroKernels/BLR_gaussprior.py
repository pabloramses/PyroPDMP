from boomerang_sampler import Boomerang

import math
import numpy as np
from collections import OrderedDict

import torch

import pyro
import pyro.distributions as dist


"Definition of the model"
true_coefs = torch.tensor([1., 2.])
data = torch.randn(2000, 2)
dim = 2
sigma = 3
labels = torch.matmul(data, true_coefs) + torch.normal(0, sigma, size = (1, 2000))
def model(data):
     coefs_mean = torch.zeros(dim)
     coefs_var = torch.ones(dim)
     sigma = 3
     coefs = pyro.sample('beta', dist.Normal(coefs_mean, coefs_var))
     y = pyro.sample('y', dist.Normal((coefs * data).sum(-1), sigma), obs=labels)
     return y


"REFERENCE MEASURE TUNING"
Sigma_ref = torch.eye(dim)
Sigma_ref_inv = Sigma_ref

"DEFINITION OF SAMPLER"
boomerang_kernel = Boomerang(model, Sigma=Sigma_ref, refresh_rate = 10, ihpp_sampler = 'Corbella') #->Corbella bounding of rates
from pyro.infer import MCMC
mcmc = MCMC(boomerang_kernel, num_samples=100)
mcmc.run(data)
print(mcmc.get_samples()['beta'].mean(0))