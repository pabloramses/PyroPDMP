from boomerang_sampler import Boomerang
import torch
import pyro

"Definition of the model asdfg"
true_coefs = torch.tensor([1., 2.])
data = torch.randn(2000, 2)
dim = 2
sigma = 0.5
labels = torch.matmul(data, true_coefs) + torch.normal(0, sigma, size = (1, 2000))
def model(data):
     coefs_mean = torch.zeros(dim)
     coefs_var = torch.ones(dim)
     sigma = 0.5
     coefs = pyro.sample('beta', dist.Normal(coefs_mean, coefs_var))
     y = pyro.sample('y', dist.Normal((coefs * data).sum(-1), sigma), obs=labels)
     return y


"REFERENCE MEASURE TUNING"
Sigma_ref = torch.eye(dim)
Sigma_ref_inv = Sigma_ref

"BOUND ON HESSIAN"
Target_sigma_inv = Sigma_ref_inv + (1/sigma) * torch.matmul(data.transpose(0,-1), data)
hessian_bound = torch.linalg.matrix_norm(Target_sigma_inv).item()

"DEFINITION OF SAMPLER"
boomerang_kernel = Boomerang(model, Sigma=Sigma_ref, hessian_bound=hessian_bound, refresh_rate = 1.0)
from pyro.infer import MCMC
mcmc = MCMC(boomerang_kernel, num_samples=1000)
mcmc.run(data)
print(mcmc.get_samples()['beta'].mean(0))