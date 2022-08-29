import math
import numpy as np
from collections import OrderedDict

import os
import torch
import torch.nn.functional as F
from pyro.infer import MCMC, NUTS, Predictive
import pyro
import pandas as pd
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
from sklearn.metrics import r2_score
from Pyro_Boomerang import Boomerang
from Pyro_Zigzag import ZZ
from Pyro_BPS import BPS
from utils import *

PATH = os.getcwd()

#True Model
dim_mod = 50
total_dim = 2 * dim_mod + 2
sample_size = 1000
"Definition of the model"
def model(data):
     coefs_mean = torch.zeros(dim_mod)
     lambdas = pyro.sample('lambdas', dist.HalfCauchy(1).expand([dim_mod]))
     tau = pyro.sample('tau', dist.HalfCauchy(1))
     sigma = pyro.sample('sigma', dist.Gamma(1,1))
     coefs = pyro.sample('beta', dist.Normal(coefs_mean, (tau * lambdas)**2))
     y = pyro.sample('y', dist.Normal((coefs * data).sum(-1), sigma), obs=labels)
     return y



num_samples = 1000
warmup_steps = 1000

r2scores_bk_d100_smed = []
sparsity_bk_d100_smed = []
perCorrect_bk_d100_smed = []
convergence_bk_d100_smed = []

r2scores_bps_d100_smed = []
sparsity_bps_d100_smed = []
perCorrect_bps_d100_smed = []
convergence_bps_d100_smed = []

r2scores_hmc_d100_smed = []
sparsity_hmc_d100_smed = []
perCorrect_hmc_d100_smed = []
convergence_hmc_d100_smed = []

r2scores_zz_d100_smed = []
sparsity_zz_d100_smed = []
perCorrect_zz_d100_smed = []
convergence_zz_d100_smed = []
import matplotlib.pyplot as plt


mu_betas = torch.randint(1,3, size=(1,dim_mod))[0]*1.0
coefs_100 = torch.normal(mean=mu_betas, std=torch.ones(dim_mod)*0.2)
sparse_coefs_100 = torch.randint(0,2,(dim_mod,))*1.0
true_coefs_100 = torch.normal(mean=mu_betas, std=torch.ones(dim_mod)) * sparse_coefs_100

"10 SAMPLES"
data_d100 = torch.randn(sample_size, dim_mod)
true_y_d100 = torch.matmul(data_d100, true_coefs_100)

"med noise SNR = 50"
SNR = 10
sigma_med = true_y_d100.var(0) / SNR
y_d50_slar = true_y_d100 + torch.normal(0, sigma_med, size=(1, sample_size))

"Tune model"
labels = y_d50_slar


#######################################BPS########################################################
"DEFINITION OF SAMPLER"
bps_d100_smed = BPS(model, dimension=total_dim, refresh_rate = 10, ihpp_sampler = 'Corbella')
mcmc_bps_d100_smed = MCMC(bps_d100_smed, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_bps_d100_smed.run(data_d100)
"posterior distribution"
postMean_bps_d100_smed = mcmc_bps_d100_smed.get_samples()['beta'].mean(0)
"get samples from predictive distribution"
postSamp_bps_d100_smed = mcmc_bps_d100_smed.get_samples()['beta']
predSamp_bps_d100_smed = predictive_samples(postSamp_bps_d100_smed, data_d100)
"SAVE TO CSV"
postSamp_bps_d100_smed_df = pd.DataFrame(postSamp_bps_d100_smed.numpy())
"summary of predictions"
predMean_bps_d100_smed ,predLower_bps_d100_smed, predUpper_bps_d100_smed = predictive_summary(predSamp_bps_d100_smed, 0.025)
print("bps r2", r2_score(labels.squeeze(), predMean_bps_d100_smed))
print("bps percentage", percentage_correct(predLower_bps_d100_smed,predUpper_bps_d100_smed,true_y_d100))
"r2 score"
r2scores_bps_d100_smed.append(r2_score(labels.squeeze(), predMean_bps_d100_smed))
perCorrect_bps_d100_smed.append(percentage_correct(predLower_bps_d100_smed,predUpper_bps_d100_smed,true_y_d100))

"Convergences"
ksd_bps_d100_smed = KSD_hp(bps_d100_smed, mcmc_bps_d100_smed.get_samples(), K=999)
convergence_bps_d100_smed.append(ksd_bps_d100_smed)
print("bps convergence", ksd_bps_d100_smed)
#######################################ZZ########################################################
"DEFINITION OF SAMPLER"
zz_d100_smed = ZZ(model, dimension=total_dim, excess_rate = 2, ihpp_sampler = 'Corbella')
mcmc_zz_d100_smed = MCMC(zz_d100_smed, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_zz_d100_smed.run(data_d100)
"posterior distribution"
postMean_zz_d100_smed = mcmc_zz_d100_smed.get_samples()['beta'].mean(0)
"get samples from predictive distribution"
postSamp_zz_d100_smed = mcmc_zz_d100_smed.get_samples()['beta']
predSamp_zz_d100_smed = predictive_samples(postSamp_zz_d100_smed, data_d100)
"SAVE TO CSV"
postSamp_zz_d100_smed_df = pd.DataFrame(postSamp_zz_d100_smed.numpy())
"summary of predictions"
predMean_zz_d100_smed ,predLower_zz_d100_smed, predUpper_zz_d100_smed = predictive_summary(predSamp_zz_d100_smed, 0.025)
print("zz r2", r2_score(labels.squeeze(), predMean_zz_d100_smed))
print("zz percentage", percentage_correct(predLower_zz_d100_smed,predUpper_zz_d100_smed,true_y_d100))
"r2 score"
r2scores_zz_d100_smed.append(r2_score(labels.squeeze(), predMean_zz_d100_smed))
perCorrect_zz_d100_smed.append(percentage_correct(predLower_zz_d100_smed,predUpper_zz_d100_smed,true_y_d100))

"Convergences"
ksd_zz_d100_smed = KSD_hp_2(zz_d100_smed, mcmc_zz_d100_smed.get_samples(), K=999)
convergence_zz_d100_smed.append(ksd_zz_d100_smed)
print("zz convergence", ksd_zz_d100_smed)
