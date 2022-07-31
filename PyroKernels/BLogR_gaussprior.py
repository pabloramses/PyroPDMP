from boomerang_sampler import Boomerang
import torch
import numpy as np
import pyro
import pyro.distributions as dist

true_coefs = torch.tensor([1., 2.])
data = torch.randn(2000, 2)
dim = 2
labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()

def model(data):
     coefs_mean = torch.zeros(dim)
     coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(2)))
     y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
     return y

boomerang_kernel = Boomerang(model, Sigma=np.array([[3,0.5],[0.5,3]]), refresh_rate = 10, ihpp_sampler='Corbella')
from pyro.infer import MCMC
mcmc = MCMC(boomerang_kernel, num_samples=100)
mcmc.run(data)
print(mcmc.get_samples()['beta'].mean(0))