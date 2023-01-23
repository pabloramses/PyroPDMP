import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
from pyro.infer import NUTS, Predictive
from MCMC import MCMC
import pyro
import pandas as pd
import pyro.distributions as dist
from pyro.distributions.testing.fakes import NonreparameterizedNormal
from pyro.distributions.util import scalar_like
from pyro.infer.autoguide import init_to_uniform, init_to_value, init_to_sample
from pyro.infer.mcmc.adaptation import WarmupAdapter
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from pyro.ops.integrator import potential_grad, velocity_verlet
from pyro.util import optional, torch_isnan
from scipy import optimize
from scipy import integrate
from sklearn.metrics import r2_score
from Pyro_Boomerang_sub import Boomerang

from utils import *
from TrajectorySample import TrajectorySample

batch_size = 100
NUM_SAMPLES = 1000
WARM_UP = 1000

sample_size = 10000
dim = 10
coefs_mean = torch.zeros(dim)

def metropolis_step(current_betas, current_lambdas, current_tau):
    lambda_proposal = pyro.sample('prop_lambda', dist.HalfCauchy(1).expand([dim]))
    tau_proposal = pyro.sample('prop_tau', dist.HalfCauchy(1))
    unif = torch.rand(dim)
    lambdas = current_lambdas
    tau = current_tau
    for i in range(dim):
        gauss_current = torch.distributions.normal.Normal(0,current_lambdas[i]*current_tau)
        gauss_prop = torch.distributions.normal.Normal(0,lambda_proposal[i]*current_tau)
        if torch.log(unif[i]) < (gauss_prop.log_prob(current_betas[i]) - gauss_current.log_prob(current_betas[i])):
            lambdas[i] = lambda_proposal[i]
    current_sum = 0
    proposal_sum = 0
    for i in range(dim):
        current_sum = current_sum + torch.distributions.normal.Normal(0,current_lambdas[i]*current_tau).log_prob(current_betas[i])
        proposal_sum = proposal_sum + torch.distributions.normal.Normal(0,current_lambdas[i]*tau_proposal).log_prob(current_betas[i])
    if torch.log(torch.rand(1))<proposal_sum - current_sum:
        tau = current_tau
    return lambdas, tau

def model(data, labels):
    lambdas = pyro.sample('lambda', dist.HalfCauchy(1).expand([dim]))
    tau = pyro.sample('tau', dist.HalfCauchy(1))
    sigma = pyro.sample('sigma', dist.Gamma(1, 1))
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, (tau * lambdas) ** 2))
    y = pyro.sample('y', dist.Normal((coefs * data).sum(-1), sigma), obs=labels)
    return y, lambdas, tau



mu_betas = torch.randint(-3,3, size=(1,10))[0]*1.0
true_coefs_10 = torch.normal(mean=mu_betas, std=torch.ones(10))

"10 SAMPLES"
data_d10_n1000 = torch.randn(sample_size, 10)
true_y_d10_n1000 = torch.matmul(data_d10_n1000, true_coefs_10)
dim = 10

"med noise SNR = 50"
SNR = 10
sigma_med = true_y_d10_n1000.var(0) / SNR
y_d10_n1000_smed = true_y_d10_n1000 + torch.normal(0, sigma_med, size = (1, sample_size))


"Tune model"
errors = y_d10_n1000_smed.transpose(0,-1) - torch.matmul(data_d10_n1000,torch.matmul(torch.matmul(torch.inverse(torch.matmul(data_d10_n1000.transpose(0,-1), data_d10_n1000)), data_d10_n1000.transpose(0,-1)), y_d10_n1000_smed.transpose(0,-1)))
sigma = torch.sqrt(torch.mean(errors**2)) #MLE estimation of noise
labels = y_d10_n1000_smed
truePost = torch.matmul(torch.inverse(torch.eye(dim) + (1/sigma**2) * torch.matmul(data_d10_n1000.transpose(0,-1),data_d10_n1000)) , (1/sigma**2) * torch.matmul(data_d10_n1000.transpose(0,-1), labels.transpose(0,-1)))

parameters = ["beta"]
hyperparameters = ["lambda", "sigma", "tau"]
init_lambdas = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample(torch.Size([dim]))
init_tau = torch.distributions.half_cauchy.HalfCauchy(scale=1).sample()
init_beta = torch.distributions.normal.Normal(0,1).sample(torch.tensor([dim]))
init_sigma = torch.distributions.gamma.Gamma(1,1).sample()
initial_values = {"beta": init_beta, "lambda":init_lambdas, "tau":init_tau, "sigma":init_sigma}

bk_d10_n1000_smed = Boomerang(model, Sigma=np.eye(10), refresh_rate=10, ihpp_sampler='Corbella', parameter_list=parameters, hyperparameter_list=hyperparameters, gibbs_rate=10, initial_parameters=initial_values, RW_scale=0.1, batch_size=10)
mcmc_bk_d10_n1000_smed = MCMC(bk_d10_n1000_smed, num_samples=NUM_SAMPLES, warmup_steps=WARM_UP)
mcmc_bk_d10_n1000_smed.run(data_d10_n1000, labels)
trajectory = TrajectorySample(bk_d10_n1000_smed, {"beta":mcmc_bk_d10_n1000_smed.get_samples()["beta"]}, 500)
samples = trajectory.sample
print("true coefs:", true_coefs_10)
print("posterior means", samples['beta'].mean(0))
#print("bk distance", torch.norm(postmean - truePost.transpose(0, -1)))