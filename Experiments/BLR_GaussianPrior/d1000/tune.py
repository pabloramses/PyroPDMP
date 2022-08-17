import math
import numpy as np
from collections import OrderedDict


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
import os

PATH = os.path.dirname(__file__)

#True Model
def model(data):
    coefs_mean = torch.zeros(dim)
    coefs_var = torch.ones(dim)
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, coefs_var))
    y = pyro.sample('y', dist.Normal((coefs * data).sum(-1), sigma), obs=labels)
    return y



sample_size = 10000
dim = 1000

num_samples = 5000
warmup_steps = 5000



r2scores_bk_d1000_n10000_smed = []
perCorrect_bk_d1000_n10000_smed = []
distances_bk_d1000_n10000_smed = []
convergence_bk_d1000_n10000_smed = []

r2scores_bps_d1000_n10000_smed = []
perCorrect_bps_d1000_n10000_smed = []
distances_bps_d1000_n10000_smed = []
convergence_bps_d1000_n10000_smed = []

r2scores_hmc_d1000_n10000_smed = []
perCorrect_hmc_d1000_n10000_smed = []
distances_hmc_d1000_n10000_smed = []
convergence_hmc_d1000_n10000_smed = []

r2scores_zz_d1000_n10000_smed = []
perCorrect_zz_d1000_n10000_smed = []
distances_zz_d1000_n10000_smed = []
convergence_zz_d1000_n10000_smed = []





mu_betas = torch.randint(-3,3, size=(1,dim))[0]*1.0
true_coefs_10 = torch.normal(mean=mu_betas, std=torch.ones(dim))

"10 SAMPLES"
data_d1000_n10000 = torch.randn(sample_size, dim)
true_y_d1000_n10000 = torch.matmul(data_d1000_n10000, true_coefs_10)

"med noise SNR = 20"
SNR = 100
sigma_med = true_y_d1000_n10000.var(0) / SNR
y_d1000_n10000_smed = true_y_d1000_n10000 + torch.normal(0, sigma_med, size = (1, sample_size))


"Tune model"
errors = y_d1000_n10000_smed.transpose(0,-1) - torch.matmul(data_d1000_n10000,torch.matmul(torch.matmul(torch.inverse(torch.matmul(data_d1000_n10000.transpose(0,-1), data_d1000_n10000)), data_d1000_n10000.transpose(0,-1)), y_d1000_n10000_smed.transpose(0,-1)))
sigma = torch.sqrt(torch.mean(errors**2)) #MLE estimation of noise
labels = y_d1000_n10000_smed
truePost = torch.matmul(torch.inverse(torch.eye(dim) + (1/sigma**2) * torch.matmul(data_d1000_n10000.transpose(0,-1),data_d1000_n10000)) , (1/sigma**2) * torch.matmul(data_d1000_n10000.transpose(0,-1), labels.transpose(0,-1)))
#################################BOOMERANG##################################
"BOUND ON HESSIAN"
Target_sigma_inv = torch.eye(dim) + (1/sigma_med**2) * torch.matmul(data_d1000_n10000.transpose(0,-1), data_d1000_n10000)
hessian_bound = torch.linalg.matrix_norm(Target_sigma_inv).item()
"REFERENCE MEASURE TUNING"
Sigma_ref = torch.eye(dim)
"DEFINITION OF SAMPLER"
bk_d1000_n10000_smed = Boomerang(model, Sigma=Sigma_ref, hessian_bound = hessian_bound, refresh_rate = 25, ihpp_sampler = 'Exact')
mcmc_bk_d1000_n10000_smed = MCMC(bk_d1000_n10000_smed, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_bk_d1000_n10000_smed.run(data_d1000_n10000)
"posterior distribution"
postMean_bk_d1000_n10000_smed = mcmc_bk_d1000_n10000_smed.get_samples()['beta'].mean(0)
"get samples from predictive distribution"
postSamp_bk_d1000_n10000_smed = mcmc_bk_d1000_n10000_smed.get_samples()['beta']
print("25 bk distance", torch.norm(postMean_bk_d1000_n10000_smed - truePost.transpose(0,-1)))
predSamp_bk_d1000_n10000_smed = predictive_samples(postSamp_bk_d1000_n10000_smed, data_d1000_n10000)
"SAVE TO CSV"
postSamp_bk_d1000_n10000_smed_df = pd.DataFrame(postSamp_bk_d1000_n10000_smed.numpy())
postSamp_bk_d1000_n10000_smed_df.to_csv(PATH + "/results/d1000_n10000_smed/postSamp_bk_d1000_n10000_smed_run"+str(i)+".csv")
"summary of predictions"
predMean_bk_d1000_n10000_smed ,predLower_bk_d1000_n10000_smed, predUpper_bk_d1000_n10000_smed = predictive_summary(predSamp_bk_d1000_n10000_smed, 0.025)
print("25 bk r2", r2_score(labels.squeeze(), predMean_bk_d1000_n10000_smed))
print("25 bk percentage", percentage_correct(predLower_bk_d1000_n10000_smed,predUpper_bk_d1000_n10000_smed,true_y_d1000_n10000))

"Scores"
r2scores_bk_d1000_n10000_smed.append(r2_score(labels.squeeze(), predMean_bk_d1000_n10000_smed))
perCorrect_bk_d1000_n10000_smed.append(percentage_correct(predLower_bk_d1000_n10000_smed,predUpper_bk_d1000_n10000_smed,true_y_d1000_n10000))
distances_bk_d1000_n10000_smed.append(torch.norm(postMean_bk_d1000_n10000_smed - truePost.transpose(0,-1)))


bk_d1000_n10000_smed = Boomerang(model, Sigma=Sigma_ref, hessian_bound = hessian_bound, refresh_rate = 10, ihpp_sampler = 'Exact')
mcmc_bk_d1000_n10000_smed = MCMC(bk_d1000_n10000_smed, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_bk_d1000_n10000_smed.run(data_d1000_n10000)
"posterior distribution"
postMean_bk_d1000_n10000_smed = mcmc_bk_d1000_n10000_smed.get_samples()['beta'].mean(0)
"get samples from predictive distribution"
postSamp_bk_d1000_n10000_smed = mcmc_bk_d1000_n10000_smed.get_samples()['beta']
print("10 bk distance", torch.norm(postMean_bk_d1000_n10000_smed - truePost.transpose(0,-1)))
predSamp_bk_d1000_n10000_smed = predictive_samples(postSamp_bk_d1000_n10000_smed, data_d1000_n10000)
"SAVE TO CSV"
postSamp_bk_d1000_n10000_smed_df = pd.DataFrame(postSamp_bk_d1000_n10000_smed.numpy())
postSamp_bk_d1000_n10000_smed_df.to_csv(PATH + "/results/d1000_n10000_smed/postSamp_bk_d1000_n10000_smed_run"+str(i)+".csv")
"summary of predictions"
predMean_bk_d1000_n10000_smed ,predLower_bk_d1000_n10000_smed, predUpper_bk_d1000_n10000_smed = predictive_summary(predSamp_bk_d1000_n10000_smed, 0.025)
print("10 bk r2", r2_score(labels.squeeze(), predMean_bk_d1000_n10000_smed))
print("10 bk percentage", percentage_correct(predLower_bk_d1000_n10000_smed,predUpper_bk_d1000_n10000_smed,true_y_d1000_n10000))

"Scores"
r2scores_bk_d1000_n10000_smed.append(r2_score(labels.squeeze(), predMean_bk_d1000_n10000_smed))
perCorrect_bk_d1000_n10000_smed.append(percentage_correct(predLower_bk_d1000_n10000_smed,predUpper_bk_d1000_n10000_smed,true_y_d1000_n10000))
distances_bk_d1000_n10000_smed.append(torch.norm(postMean_bk_d1000_n10000_smed - truePost.transpose(0,-1)))

bk_d1000_n10000_smed = Boomerang(model, Sigma=Sigma_ref, hessian_bound = hessian_bound, refresh_rate = 20, ihpp_sampler = 'Exact')
mcmc_bk_d1000_n10000_smed = MCMC(bk_d1000_n10000_smed, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_bk_d1000_n10000_smed.run(data_d1000_n10000)
"posterior distribution"
postMean_bk_d1000_n10000_smed = mcmc_bk_d1000_n10000_smed.get_samples()['beta'].mean(0)
"get samples from predictive distribution"
postSamp_bk_d1000_n10000_smed = mcmc_bk_d1000_n10000_smed.get_samples()['beta']
print("20 bk distance", torch.norm(postMean_bk_d1000_n10000_smed - truePost.transpose(0,-1)))
predSamp_bk_d1000_n10000_smed = predictive_samples(postSamp_bk_d1000_n10000_smed, data_d1000_n10000)
"SAVE TO CSV"
postSamp_bk_d1000_n10000_smed_df = pd.DataFrame(postSamp_bk_d1000_n10000_smed.numpy())
postSamp_bk_d1000_n10000_smed_df.to_csv(PATH + "/results/d1000_n10000_smed/postSamp_bk_d1000_n10000_smed_run"+str(i)+".csv")
"summary of predictions"
predMean_bk_d1000_n10000_smed ,predLower_bk_d1000_n10000_smed, predUpper_bk_d1000_n10000_smed = predictive_summary(predSamp_bk_d1000_n10000_smed, 0.025)
print("20 bk r2", r2_score(labels.squeeze(), predMean_bk_d1000_n10000_smed))
print("20 bk percentage", percentage_correct(predLower_bk_d1000_n10000_smed,predUpper_bk_d1000_n10000_smed,true_y_d1000_n10000))

"Scores"
r2scores_bk_d1000_n10000_smed.append(r2_score(labels.squeeze(), predMean_bk_d1000_n10000_smed))
perCorrect_bk_d1000_n10000_smed.append(percentage_correct(predLower_bk_d1000_n10000_smed,predUpper_bk_d1000_n10000_smed,true_y_d1000_n10000))
distances_bk_d1000_n10000_smed.append(torch.norm(postMean_bk_d1000_n10000_smed - truePost.transpose(0,-1)))

bk_d1000_n10000_smed = Boomerang(model, Sigma=Sigma_ref, hessian_bound = hessian_bound, refresh_rate = 5, ihpp_sampler = 'Exact')
mcmc_bk_d1000_n10000_smed = MCMC(bk_d1000_n10000_smed, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_bk_d1000_n10000_smed.run(data_d1000_n10000)
"posterior distribution"
postMean_bk_d1000_n10000_smed = mcmc_bk_d1000_n10000_smed.get_samples()['beta'].mean(0)
"get samples from predictive distribution"
postSamp_bk_d1000_n10000_smed = mcmc_bk_d1000_n10000_smed.get_samples()['beta']
print("5 bk distance", torch.norm(postMean_bk_d1000_n10000_smed - truePost.transpose(0,-1)))
predSamp_bk_d1000_n10000_smed = predictive_samples(postSamp_bk_d1000_n10000_smed, data_d1000_n10000)
"SAVE TO CSV"
postSamp_bk_d1000_n10000_smed_df = pd.DataFrame(postSamp_bk_d1000_n10000_smed.numpy())
postSamp_bk_d1000_n10000_smed_df.to_csv(PATH + "/results/d1000_n10000_smed/postSamp_bk_d1000_n10000_smed_run"+str(i)+".csv")
"summary of predictions"
predMean_bk_d1000_n10000_smed ,predLower_bk_d1000_n10000_smed, predUpper_bk_d1000_n10000_smed = predictive_summary(predSamp_bk_d1000_n10000_smed, 0.025)
print("5 bk r2", r2_score(labels.squeeze(), predMean_bk_d1000_n10000_smed))
print("5 bk percentage", percentage_correct(predLower_bk_d1000_n10000_smed,predUpper_bk_d1000_n10000_smed,true_y_d1000_n10000))

"Scores"
r2scores_bk_d1000_n10000_smed.append(r2_score(labels.squeeze(), predMean_bk_d1000_n10000_smed))
perCorrect_bk_d1000_n10000_smed.append(percentage_correct(predLower_bk_d1000_n10000_smed,predUpper_bk_d1000_n10000_smed,true_y_d1000_n10000))
distances_bk_d1000_n10000_smed.append(torch.norm(postMean_bk_d1000_n10000_smed - truePost.transpose(0,-1)))

"to pandas bk"
r2scores_bk_d1000_n10000_smed_df = pd.DataFrame(r2scores_bk_d1000_n10000_smed)
perCorrect_bk_d1000_n10000_smed_df = pd.DataFrame(perCorrect_bk_d1000_n10000_smed)
distances_bk_d1000_n10000_smed_df = pd.DataFrame(distances_bk_d1000_n10000_smed)
"to csv bps"
r2scores_bk_d1000_n10000_smed_df.to_csv(PATH + "/results/d1000_n10000_smed/r2scores_bk_d1000_n10000_smed.csv")
perCorrect_bk_d1000_n10000_smed_df.to_csv(PATH + "/results/d1000_n10000_smed/perCorrect_bk_d1000_n10000_smed.csv")
distances_bk_d1000_n10000_smed_df.to_csv(PATH + "/results/d1000_n10000_smed/distances_bk_d1000_n10000_smed.csv")