import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
from pyro.infer import MCMC, NUTS, Predictive
import pyro
import pyro.distributions as dist
from pyro.distributions.testing.fakes import NonreparameterizedNormal
from pyro.distributions.util import scalar_like
from pyro.infer.autoguide import init_to_uniform
from pyro.infer.mcmc.adaptation import WarmupAdapter
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from pyro.ops.integrator import potential_grad, velocity_verlet
from pyro.util import optional, torch_isnan
from scipy import optimize
from scipy import integrate

from Pyro_Boomerang import Boomerang


def model(data):
    coefs_mean = torch.zeros(dim)
    coefs_var = torch.ones(dim)
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, coefs_var))
    y = pyro.sample('y', dist.Normal((coefs * data).sum(-1), sigma), obs=labels)
    return y
dim = 1000
mu_betas = torch.randint(2,5, size=(1,dim))[0]*1.0
true_coefs_1000 = torch.normal(mean=mu_betas, std=torch.ones(dim))


data_d1000_n10000 = torch.randn(10000,dim)
true_y_d1000_n10000 = torch.matmul(data_d1000_n10000, true_coefs_1000)

SNR = 10000
sigma_low_1000 = true_y_d1000_n10000.var(0) / SNR
y_d1000_n10000_slow = true_y_d1000_n10000 + torch.normal(0, sigma_low_1000, size = (1, 10000))

"MODEL FIT FOR SCENARIO"
sigma = sigma_low_1000
labels = y_d1000_n10000_slow
"BOUND ON HESSIAN"
Target_sigma_inv = torch.eye(dim) + (1/sigma_low_1000**2) * torch.matmul(data_d1000_n10000.transpose(0,-1), data_d1000_n10000)
Target_sigma = torch.inverse(Target_sigma_inv)
hessian_bound = torch.linalg.matrix_norm(Target_sigma_inv).item()
"REFERENCE MEASURE TUNING"
Sigma_ref = torch.eye(dim)
"DEFINITION OF SAMPLER"
#boomerang_kernel = Boomerang(model, Sigma=Sigma_ref, refresh_rate = 10, ihpp_sampler = 'Corbella')
#mcmc = MCMC(boomerang_kernel, num_samples=10000, warmup_steps=1000)
#mcmc.run(data)
#print(mcmc.get_samples()['beta'].mean(0))
bk_d1000_n10000 = Boomerang(model, Sigma=Sigma_ref, hessian_bound = hessian_bound, refresh_rate = 10, ihpp_sampler = 'Exact')
mcmc_bk_d1000_n10000_slow = MCMC(bk_d1000_n10000, num_samples=10000, warmup_steps=2000)
mcmc_bk_d1000_n10000_slow.run(data_d1000_n10000)