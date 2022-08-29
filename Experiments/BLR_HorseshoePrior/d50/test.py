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
K=1000

r2scores_bk_d50_smed = []
sparsity_bk_d50_smed = []
perCorrect_bk_d50_smed = []
convergence_bk_d50_smed = []

r2scores_bps_d50_smed = []
sparsity_bps_d50_smed = []
perCorrect_bps_d50_smed = []
convergence_bps_d50_smed = []

r2scores_hmc_d50_smed = []
sparsity_hmc_d50_smed = []
perCorrect_hmc_d50_smed = []
convergence_hmc_d50_smed = []

r2scores_zz_d50_smed = []
sparsity_zz_d50_smed = []
perCorrect_zz_d50_smed = []
convergence_zz_d50_smed = []
import matplotlib.pyplot as plt


mu_betas = torch.randint(1,3, size=(1,dim_mod))[0]*1.0
coefs_100 = torch.normal(mean=mu_betas, std=torch.ones(dim_mod)*0.2)
sparse_coefs_100 = torch.randint(0,2,(dim_mod,))*1.0
true_coefs_100 = coefs_100 * sparse_coefs_100

"10 SAMPLES"
data_d50 = torch.randn(sample_size, dim_mod)
true_y_d50 = torch.matmul(data_d50, true_coefs_100)

"med noise SNR = 50"
SNR = 10
sigma_med = true_y_d50.var(0) / SNR
y_d50_smed = true_y_d50 + torch.normal(0, sigma_med, size=(1, sample_size))

"Tune model"
labels = y_d50_smed

"""
#################################BOOMERANG##################################
"REFERENCE MEASURE TUNING"
Sigma_ref = torch.eye(total_dim)
"DEFINITION OF SAMPLER"
bk_d50_smed = Boomerang(model, Sigma=Sigma_ref, refresh_rate=10, ihpp_sampler='Corbella')
mcmc_bk_d50_smed = MCMC(bk_d50_smed, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_bk_d50_smed.run(data_d50)
"posterior distribution"
postMean_bk_d50_smed = mcmc_bk_d50_smed.get_samples()['beta'].mean(0)
print(postMean_bk_d50_smed.shape)
sparsity_bk_d50_smed_i = sparsity(sparse_coefs_100, postMean_bk_d50_smed)
print("bk sparsity", sparsity_bk_d50_smed_i)
"get samples from predictive distribution"
postSamp_bk_d50_smed = mcmc_bk_d50_smed.get_samples()['beta']
predSamp_bk_d50_smed = predictive_samples(postSamp_bk_d50_smed, data_d50)
"SAVE TO CSV"
postSamp_bk_d50_smed_df = pd.DataFrame(postSamp_bk_d50_smed.numpy())
"summary of predictions"
predMean_bk_d50_smed, predLower_bk_d50_smed, predUpper_bk_d50_smed = predictive_summary(predSamp_bk_d50_smed,0.025)
print("bk r2", r2_score(labels.squeeze(), predMean_bk_d50_smed))
print("bk percentage", percentage_correct(predLower_bk_d50_smed, predUpper_bk_d50_smed, true_y_d50))

"Convergences"
ksd_bk_d50_smed_1 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=10, beta=0.1,  K=K)
ksd_bk_d50_smed_2 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=10, beta=0.25, K=K)
ksd_bk_d50_smed_3 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=10, beta=0.5, K=K)
ksd_bk_d50_smed_4 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=10, beta=0.75, K=K)
ksd_bk_d50_smed_5 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=10, beta=0.9, K=K)
ksd_bk_d50_smed_6 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=100, beta=0.1, K=K)
ksd_bk_d50_smed_7 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=100, beta=0.25, K=K)
ksd_bk_d50_smed_8 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=100, beta=0.5, K=K)
ksd_bk_d50_smed_9 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=100, beta=0.75, K=K)
ksd_bk_d50_smed_10 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=100, beta=0.9, K=K)
ksd_bk_d50_smed_11 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=1000, beta=0.1, K=K)
ksd_bk_d50_smed_12 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=1000, beta=0.25, K=K)
ksd_bk_d50_smed_13= KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=1000, beta=0.5, K=K)
ksd_bk_d50_smed_14= KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=1000, beta=0.75, K=K)
ksd_bk_d50_smed_15= KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=1000, beta=0.9, K=K)
ksd_bk_d50_smed_16= KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=10000, beta=0.1, K=K)
ksd_bk_d50_smed_17= KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=10000, beta=0.25, K=K)
ksd_bk_d50_smed_18= KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=10000, beta=0.5, K=K)
ksd_bk_d50_smed_19= KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=10000, beta=0.75, K=K)
ksd_bk_d50_smed_20= KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=10000, beta=0.9, K=K)
ksd_bk_d50_smed_21 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=100000, beta=0.1, K=K)
ksd_bk_d50_smed_22 = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=100000, beta=0.25, K=K)
ksd_bk_d50_smed_23= KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=100000, beta=0.5, K=K)
ksd_bk_d50_smed_24= KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=100000, beta=0.75, K=K)
ksd_bk_d50_smed_25= KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(), c=100000, beta=0.9, K=K)

print("bk convergence 1", ksd_bk_d50_smed_1)
print("bk convergence 2", ksd_bk_d50_smed_2)
print("bk convergence 3", ksd_bk_d50_smed_3)
print("bk convergence 4", ksd_bk_d50_smed_4)
print("bk convergence 5", ksd_bk_d50_smed_5)
print("bk convergence 6", ksd_bk_d50_smed_6)
print("bk convergence 7", ksd_bk_d50_smed_7)
print("bk convergence 8", ksd_bk_d50_smed_8)
print("bk convergence 9", ksd_bk_d50_smed_9)
print("bk convergence 10", ksd_bk_d50_smed_10)
print("bk convergence 11", ksd_bk_d50_smed_11)
print("bk convergence 12", ksd_bk_d50_smed_12)
print("bk convergence 13", ksd_bk_d50_smed_13)
print("bk convergence 14", ksd_bk_d50_smed_14)
print("bk convergence 15", ksd_bk_d50_smed_15)
print("bk convergence 16", ksd_bk_d50_smed_16)
print("bk convergence 17", ksd_bk_d50_smed_17)
print("bk convergence 18", ksd_bk_d50_smed_18)
print("bk convergence 19", ksd_bk_d50_smed_19)
print("bk convergence 20", ksd_bk_d50_smed_20)
print("bk convergence 21", ksd_bk_d50_smed_21)
print("bk convergence 22", ksd_bk_d50_smed_22)
print("bk convergence 23", ksd_bk_d50_smed_23)
print("bk convergence 24", ksd_bk_d50_smed_24)
print("bk convergence 25", ksd_bk_d50_smed_25)

"""

#######################################BPS########################################################
#"DEFINITION OF SAMPLER"
#bps_d50_smed = BPS(model, dimension=total_dim, refresh_rate = 10, ihpp_sampler = 'Corbella')
#mcmc_bps_d50_smed = MCMC(bps_d50_smed, num_samples=num_samples, warmup_steps=warmup_steps)
#mcmc_bps_d50_smed.run(data_d50)
#"posterior distribution"
#postMean_bps_d50_smed = mcmc_bps_d50_smed.get_samples()['beta'].mean(0)
#sparsity_bps_d50_smed_i = sparsity(sparse_coefs_100, postMean_bps_d50_smed)
#print("bk sparsity", sparsity_bps_d50_smed_i)
#"get samples from predictive distribution"
#postSamp_bps_d50_smed = mcmc_bps_d50_smed.get_samples()['beta']
#predSamp_bps_d50_smed = predictive_samples(postSamp_bps_d50_smed, data_d50)
#"SAVE TO CSV"
#postSamp_bps_d50_smed_df = pd.DataFrame(postSamp_bps_d50_smed.numpy())

#"summary of predictions"
#predMean_bps_d50_smed ,predLower_bps_d50_smed, predUpper_bps_d50_smed = predictive_summary(predSamp_bps_d50_smed, 0.025)
#print("bps r2", r2_score(labels.squeeze(), predMean_bps_d50_smed))
#print("bps percentage", percentage_correct(predLower_bps_d50_smed,predUpper_bps_d50_smed,true_y_d50))
#"r2 score"
#r2scores_bps_d50_smed.append(r2_score(labels.squeeze(), predMean_bps_d50_smed))
#perCorrect_bps_d50_smed.append(percentage_correct(predLower_bps_d50_smed,predUpper_bps_d50_smed,true_y_d50))

#"Convergences"
#ksd_bps_d50_smed = KSD_hp_2(bps_d50_smed, mcmc_bps_d50_smed.get_samples(), K=K)
#convergence_bps_d50_smed.append(ksd_bps_d50_smed)
#print("bps convergence", ksd_bps_d50_smed)
#######################################ZZ########################################################
#"DEFINITION OF SAMPLER"
#zz_d50_smed = ZZ(model, dimension=total_dim, excess_rate = 1, ihpp_sampler = 'Corbella')
#mcmc_zz_d50_smed = MCMC(zz_d50_smed, num_samples=num_samples, warmup_steps=warmup_steps)
#mcmc_zz_d50_smed.run(data_d50)
#"posterior distribution"
#postMean_zz_d50_smed = mcmc_zz_d50_smed.get_samples()['beta'].mean(0)
#sparsity_zz_d50_smed_i = sparsity(sparse_coefs_100, postMean_zz_d50_smed)
#print("bk sparsity", sparsity_zz_d50_smed_i)
#"get samples from predictive distribution"
#postSamp_zz_d50_smed = mcmc_zz_d50_smed.get_samples()['beta']
#predSamp_zz_d50_smed = predictive_samples(postSamp_zz_d50_smed, data_d50)
#"SAVE TO CSV"
#postSamp_zz_d50_smed_df = pd.DataFrame(postSamp_zz_d50_smed.numpy())
#"summary of predictions"
#predMean_zz_d50_smed ,predLower_zz_d50_smed, predUpper_zz_d50_smed = predictive_summary(predSamp_zz_d50_smed, 0.025)
#print("zz r2", r2_score(labels.squeeze(), predMean_zz_d50_smed))
#print("zz percentage", percentage_correct(predLower_zz_d50_smed,predUpper_zz_d50_smed,true_y_d50))
"r2 score"
#r2scores_zz_d50_smed.append(r2_score(labels.squeeze(), predMean_zz_d50_smed))
#perCorrect_zz_d50_smed.append(percentage_correct(predLower_zz_d50_smed,predUpper_zz_d50_smed,true_y_d50))

"Convergences"
#ksd_zz_d50_smed = KSD_hp_2(zz_d50_smed, mcmc_zz_d50_smed.get_samples(), K=999)
#convergence_zz_d50_smed.append(ksd_zz_d50_smed)
#print("zz convergence", ksd_zz_d50_smed)

#######################################HMC########################################################
"""
"DEFINITION OF SAMPLER"
hmc_d50_smed = NUTS(model)
mcmc_hmc_d50_smed = MCMC(hmc_d50_smed, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_hmc_d50_smed.run(data_d50)
"posterior distribution"
postMean_hmc_d50_smed = mcmc_hmc_d50_smed.get_samples()['beta'].mean(0)
sparsity_hmc_d50_smed_i = sparsity(sparse_coefs_100, postMean_hmc_d50_smed)
print("hmc sparsity", sparsity_hmc_d50_smed_i)
"get samples from predictive distribution"
postSamp_hmc_d50_smed = mcmc_hmc_d50_smed.get_samples()['beta']
predSamp_hmc_d50_smed = predictive_samples(postSamp_hmc_d50_smed, data_d50)
"SAVE TO CSV"
postSamp_hmc_d50_smed_df = pd.DataFrame(postSamp_hmc_d50_smed.numpy())
"summary of predictions"
predMean_hmc_d50_smed, predLower_hmc_d50_smed, predUpper_hmc_d50_smed = predictive_summary(predSamp_hmc_d50_smed, 0.025)
print("hmc r2", r2_score(labels.squeeze(), predMean_hmc_d50_smed))
print("hmc percentage", percentage_correct(predLower_hmc_d50_smed, predUpper_hmc_d50_smed, true_y_d50))
"r2 score"

"Convergences"
"Convergences"
ksd_hmc_d50_smed_1 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=10, beta=0.1,  K=K)
ksd_hmc_d50_smed_2 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=10, beta=0.25, K=K)
ksd_hmc_d50_smed_3 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=10, beta=0.5, K=K)
ksd_hmc_d50_smed_4 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=10, beta=0.75, K=K)
ksd_hmc_d50_smed_5 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=10, beta=0.9, K=K)
ksd_hmc_d50_smed_6 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=100, beta=0.1, K=K)
ksd_hmc_d50_smed_7 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=100, beta=0.25, K=K)
ksd_hmc_d50_smed_8 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=100, beta=0.5, K=K)
ksd_hmc_d50_smed_9 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=100, beta=0.75, K=K)
ksd_hmc_d50_smed_10 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=100, beta=0.9, K=K)
ksd_hmc_d50_smed_11 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=1000, beta=0.1, K=K)
ksd_hmc_d50_smed_12 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=1000, beta=0.25, K=K)
ksd_hmc_d50_smed_13= KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=1000, beta=0.5, K=K)
ksd_hmc_d50_smed_14= KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=1000, beta=0.75, K=K)
ksd_hmc_d50_smed_15= KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=1000, beta=0.9, K=K)
ksd_hmc_d50_smed_16= KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=10000, beta=0.1, K=K)
ksd_hmc_d50_smed_17= KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=10000, beta=0.25, K=K)
ksd_hmc_d50_smed_18= KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=10000, beta=0.5, K=K)
ksd_hmc_d50_smed_19= KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=10000, beta=0.75, K=K)
ksd_hmc_d50_smed_20= KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=10000, beta=0.9, K=K)
ksd_hmc_d50_smed_21 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=100000, beta=0.1, K=K)
ksd_hmc_d50_smed_22 = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=100000, beta=0.25, K=K)
ksd_hmc_d50_smed_23= KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=100000, beta=0.5, K=K)
ksd_hmc_d50_smed_24= KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=100000, beta=0.75, K=K)
ksd_hmc_d50_smed_25= KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(), c=100000, beta=0.9, K=K)

print("hmc convergence 1", ksd_hmc_d50_smed_1)
print("hmc convergence 2", ksd_hmc_d50_smed_2)
print("hmc convergence 3", ksd_hmc_d50_smed_3)
print("hmc convergence 4", ksd_hmc_d50_smed_4)
print("hmc convergence 5", ksd_hmc_d50_smed_5)
print("hmc convergence 6", ksd_hmc_d50_smed_6)
print("hmc convergence 7", ksd_hmc_d50_smed_7)
print("hmc convergence 8", ksd_hmc_d50_smed_8)
print("hmc convergence 9", ksd_hmc_d50_smed_9)
print("hmc convergence 10", ksd_hmc_d50_smed_10)
print("hmc convergence 11", ksd_hmc_d50_smed_11)
print("hmc convergence 12", ksd_hmc_d50_smed_12)
print("hmc convergence 13", ksd_hmc_d50_smed_13)
print("hmc convergence 14", ksd_hmc_d50_smed_14)
print("hmc convergence 15", ksd_hmc_d50_smed_15)
print("hmc convergence 16", ksd_hmc_d50_smed_16)
print("hmc convergence 17", ksd_hmc_d50_smed_17)
print("hmc convergence 18", ksd_hmc_d50_smed_18)
print("hmc convergence 19", ksd_hmc_d50_smed_19)
print("hmc convergence 20", ksd_hmc_d50_smed_20)
print("hmc convergence 21", ksd_hmc_d50_smed_21)
print("hmc convergence 22", ksd_hmc_d50_smed_22)
print("hmc convergence 23", ksd_hmc_d50_smed_23)
print("hmc convergence 24", ksd_hmc_d50_smed_24)
print("hmc convergence 25", ksd_hmc_d50_smed_25)

#######################################ZZ########################################################
"DEFINITION OF SAMPLER"
zz_d50_smed = ZZ(model, dimension=total_dim, excess_rate = 1, ihpp_sampler = 'Corbella')
mcmc_zz_d50_smed = MCMC(zz_d50_smed, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_zz_d50_smed.run(data_d50)
"posterior distribution"
postMean_zz_d50_smed = mcmc_zz_d50_smed.get_samples()['beta'].mean(0)
sparsity_zz_d50_smed_i = sparsity(sparse_coefs_100, postMean_zz_d50_smed)
print("zz sparsity 1", sparsity_zz_d50_smed_i)
"get samples from predictive distribution"
postSamp_zz_d50_smed = mcmc_zz_d50_smed.get_samples()['beta']
predSamp_zz_d50_smed = predictive_samples(postSamp_zz_d50_smed, data_d50)
"SAVE TO CSV"
postSamp_zz_d50_smed_df = pd.DataFrame(postSamp_zz_d50_smed.numpy())
"summary of predictions"
predMean_zz_d50_smed ,predLower_zz_d50_smed, predUpper_zz_d50_smed = predictive_summary(predSamp_zz_d50_smed, 0.025)
print("zz r2 1", r2_score(labels.squeeze(), predMean_zz_d50_smed))
print("zz percentage 1", percentage_correct(predLower_zz_d50_smed,predUpper_zz_d50_smed,true_y_d50))
"r2 score"
r2scores_zz_d50_smed.append(r2_score(labels.squeeze(), predMean_zz_d50_smed))
perCorrect_zz_d50_smed.append(percentage_correct(predLower_zz_d50_smed,predUpper_zz_d50_smed,true_y_d50))

"Convergences"
ksd_zz_d50_smed = KSD_hp_2(zz_d50_smed, mcmc_zz_d50_smed.get_samples(), K=K)
convergence_zz_d50_smed.append(ksd_zz_d50_smed)
print("zz convergence 1", ksd_zz_d50_smed)
"""
#######################################ZZ########################################################
"DEFINITION OF SAMPLER"
zz_d50_smed = ZZ(model, dimension=total_dim, excess_rate = 0.1, ihpp_sampler = 'Corbella')
mcmc_zz_d50_smed = MCMC(zz_d50_smed, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_zz_d50_smed.run(data_d50)
"posterior distribution"
postMean_zz_d50_smed = mcmc_zz_d50_smed.get_samples()['beta'].mean(0)
sparsity_zz_d50_smed_i = sparsity(sparse_coefs_100, postMean_zz_d50_smed)
print("zz sparsity 0.5", sparsity_zz_d50_smed_i)
"get samples from predictive distribution"
postSamp_zz_d50_smed = mcmc_zz_d50_smed.get_samples()['beta']
predSamp_zz_d50_smed = predictive_samples(postSamp_zz_d50_smed, data_d50)
"SAVE TO CSV"
postSamp_zz_d50_smed_df = pd.DataFrame(postSamp_zz_d50_smed.numpy())
"summary of predictions"
predMean_zz_d50_smed ,predLower_zz_d50_smed, predUpper_zz_d50_smed = predictive_summary(predSamp_zz_d50_smed, 0.025)
print("zz r2 0.5", r2_score(labels.squeeze(), predMean_zz_d50_smed))
print("zz percentage 0.5", percentage_correct(predLower_zz_d50_smed,predUpper_zz_d50_smed,true_y_d50))
print("width", torch.mean(torch.abs(predUpper_zz_d50_smed-predLower_zz_d50_smed)))
"r2 score"
r2scores_zz_d50_smed.append(r2_score(labels.squeeze(), predMean_zz_d50_smed))
perCorrect_zz_d50_smed.append(percentage_correct(predLower_zz_d50_smed,predUpper_zz_d50_smed,true_y_d50))

"Convergences"
ksd_zz_d50_smed = KSD_hp_2(zz_d50_smed, mcmc_zz_d50_smed.get_samples(), K=K)
convergence_zz_d50_smed.append(ksd_zz_d50_smed)
print("zz convergence 0.5", ksd_zz_d50_smed)
"""
#######################################ZZ########################################################
"DEFINITION OF SAMPLER"
zz_d50_smed = ZZ(model, dimension=total_dim, excess_rate = 0.25, ihpp_sampler = 'Corbella')
mcmc_zz_d50_smed = MCMC(zz_d50_smed, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_zz_d50_smed.run(data_d50)
"posterior distribution"
postMean_zz_d50_smed = mcmc_zz_d50_smed.get_samples()['beta'].mean(0)
sparsity_zz_d50_smed_i = sparsity(sparse_coefs_100, postMean_zz_d50_smed)
print("zz sparsity 0.25", sparsity_zz_d50_smed_i)
"get samples from predictive distribution"
postSamp_zz_d50_smed = mcmc_zz_d50_smed.get_samples()['beta']
predSamp_zz_d50_smed = predictive_samples(postSamp_zz_d50_smed, data_d50)
"SAVE TO CSV"
postSamp_zz_d50_smed_df = pd.DataFrame(postSamp_zz_d50_smed.numpy())
"summary of predictions"
predMean_zz_d50_smed ,predLower_zz_d50_smed, predUpper_zz_d50_smed = predictive_summary(predSamp_zz_d50_smed, 0.025)
print("zz r2 0.25", r2_score(labels.squeeze(), predMean_zz_d50_smed))
print("zz percentage 0.25", percentage_correct(predLower_zz_d50_smed,predUpper_zz_d50_smed,true_y_d50))
"r2 score"
r2scores_zz_d50_smed.append(r2_score(labels.squeeze(), predMean_zz_d50_smed))
perCorrect_zz_d50_smed.append(percentage_correct(predLower_zz_d50_smed,predUpper_zz_d50_smed,true_y_d50))

"Convergences"
ksd_zz_d50_smed = KSD_hp_2(zz_d50_smed, mcmc_zz_d50_smed.get_samples(), K=K)
convergence_zz_d50_smed.append(ksd_zz_d50_smed)
print("zz convergence 0.25", ksd_zz_d50_smed)
"""