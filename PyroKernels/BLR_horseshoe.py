"Definition of the model"
from boomerang_sampler import Boomerang
import torch
import pyro
"Still missing to randomize sigma"
true_coefs = torch.tensor([0., 1., 0., 0., 1., 1., 1., 1., 0.])
data = torch.randn(1000, 9)
dim = 9
sigma = 0.2
labels = torch.matmul(data, true_coefs) #+ torch.normal(0, sigma, size = (1, 100))
def model(data):
     coefs_mean = torch.zeros(dim)
     lambdas = pyro.sample('lambdas', dist.HalfCauchy(1).expand([9]).to_event(1))
     tau = pyro.sample('tau', dist.HalfCauchy(1))
     coefs = pyro.sample('beta', dist.Normal(coefs_mean, (tau * lambdas)**2).to_event(1))
     y = pyro.sample('y', dist.Normal((coefs * data).sum(-1), sigma), obs=labels)
     return y


"REFERENCE MEASURE TUNING"
Sigma_ref = torch.eye(19) * sigma
Sigma_ref_inv = Sigma_ref

"DEFINITION OF SAMPLER"
boomerang_kernel = Boomerang(model, Sigma=Sigma_ref, refresh_rate = 3, ihpp_sampler = 'Corbella')
from pyro.infer import MCMC
mcmc = MCMC(boomerang_kernel, num_samples=1000, warmup_steps=100)
mcmc.run(data)