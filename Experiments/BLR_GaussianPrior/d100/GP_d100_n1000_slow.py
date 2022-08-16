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

PATH = os.path.dirname(__file__)
#True Model
def model(data):
    coefs_mean = torch.zeros(dim)
    coefs_var = torch.ones(dim)
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, coefs_var))
    y = pyro.sample('y', dist.Normal((coefs * data).sum(-1), sigma), obs=labels)
    return y


sample_size = 1000
dim = 100

num_samples = 5000
warmup_steps = 5000



r2scores_bk_d100_n1000_slow = []
perCorrect_bk_d100_n1000_slow = []
distances_bk_d100_n1000_slow = []
convergence_bk_d100_n1000_slow = []

r2scores_bps_d100_n1000_slow = []
perCorrect_bps_d100_n1000_slow = []
distances_bps_d100_n1000_slow = []
convergence_bps_d100_n1000_slow = []

r2scores_hmc_d100_n1000_slow = []
perCorrect_hmc_d100_n1000_slow = []
distances_hmc_d100_n1000_slow = []
convergence_hmc_d100_n1000_slow = []

r2scores_zz_d100_n1000_slow = []
perCorrect_zz_d100_n1000_slow = []
distances_zz_d100_n1000_slow = []
convergence_zz_d100_n1000_slow = []

for i in range(10):
    mu_betas = torch.randint(-1,1, size=(1,dim))[0]*1.0
    true_coefs_100 = torch.normal(mean=mu_betas, std=torch.ones(dim))

    "10 SAMPLES"
    data_d100_n1000 = torch.randn(sample_size, dim)
    true_y_d100_n1000 = torch.matmul(data_d100_n1000, true_coefs_100)

    "Low noise SNR = 100"
    SNR = 100
    sigma_low = true_y_d100_n1000.var(0) / SNR
    y_d100_n1000_slow = true_y_d100_n1000 + torch.normal(0, sigma_low, size = (1, sample_size))


    "Tune model"
    errors = y_d100_n1000_slow.transpose(0,-1) - torch.matmul(data_d100_n1000,torch.matmul(torch.matmul(torch.inverse(torch.matmul(data_d100_n1000.transpose(0,-1), data_d100_n1000)), data_d100_n1000.transpose(0,-1)), y_d100_n1000_slow.transpose(0,-1)))
    sigma = torch.sqrt(torch.mean(errors**2)) #MLE estimation of noise
    labels = y_d100_n1000_slow
    truePost = torch.matmul(torch.inverse(torch.eye(dim) + (1/sigma**2) * torch.matmul(data_d100_n1000.transpose(0,-1),data_d100_n1000)) , (1/sigma**2) * torch.matmul(data_d100_n1000.transpose(0,-1), labels.transpose(0,-1)))
    #################################BOOMERANG##################################
    "BOUND ON HESSIAN"
    Target_sigma_inv = torch.eye(dim) + (1/sigma**2) * torch.matmul(data_d100_n1000.transpose(0,-1), data_d100_n1000)
    hessian_bound = torch.linalg.matrix_norm(Target_sigma_inv).item()
    "REFERENCE MEASURE TUNING"
    Sigma_ref = torch.eye(dim)
    "DEFINITION OF SAMPLER"
    bk_d100_n1000_slow = Boomerang(model, Sigma=Sigma_ref, hessian_bound = hessian_bound, refresh_rate = 10, ihpp_sampler = 'Exact')
    mcmc_bk_d100_n1000_slow = MCMC(bk_d100_n1000_slow, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_bk_d100_n1000_slow.run(data_d100_n1000)
    "posterior distribution"
    postMean_bk_d100_n1000_slow = mcmc_bk_d100_n1000_slow.get_samples()['beta'].mean(0)
    "get samples from predictive distribution"
    postSamp_bk_d100_n1000_slow = mcmc_bk_d100_n1000_slow.get_samples()['beta']
    predSamp_bk_d100_n1000_slow = predictive_samples(postSamp_bk_d100_n1000_slow, data_d100_n1000)
    "SAVE TO CSV"
    postSamp_bk_d100_n1000_slow_df = pd.DataFrame(postSamp_bk_d100_n1000_slow.numpy())
    postSamp_bk_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/postSamp_bk_d100_n1000_slow_run"+str(i)+".csv")
    "summary of predictions"
    predMean_bk_d100_n1000_slow ,predLower_bk_d100_n1000_slow, predUpper_bk_d100_n1000_slow = predictive_summary(predSamp_bk_d100_n1000_slow, 0.025)
    print("bk r2", r2_score(labels.squeeze(), predMean_bk_d100_n1000_slow))
    print("bk distance", torch.norm(postMean_bk_d100_n1000_slow - truePost.transpose(0,-1)))
    print("bk percentage", percentage_correct(predLower_bk_d100_n1000_slow,predUpper_bk_d100_n1000_slow,true_y_d100_n1000))
    "Scores"
    r2scores_bk_d100_n1000_slow.append(r2_score(labels.squeeze(), predMean_bk_d100_n1000_slow))
    perCorrect_bk_d100_n1000_slow.append(percentage_correct(predLower_bk_d100_n1000_slow,predUpper_bk_d100_n1000_slow,true_y_d100_n1000))
    distances_bk_d100_n1000_slow.append(torch.norm(postMean_bk_d100_n1000_slow - truePost.transpose(0,-1)))

    "Convergences"
    #k_bk_d100_n1000_slow = kernel_Stein_Discrepancies(bk_d100_n1000_slow, postSamp_bk_d100_n1000_slow)
    #convergence_bk_d100_n1000_slow.append(k_bk_d100_n1000_slow)

    #######################################BPS########################################################
    "DEFINITION OF SAMPLER"
    bps_d100_n1000_slow = BPS(model, dimension=dim, hessian_bound = hessian_bound, refresh_rate = 10, ihpp_sampler = 'Exact')
    mcmc_bps_d100_n1000_slow = MCMC(bps_d100_n1000_slow, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_bps_d100_n1000_slow.run(data_d100_n1000)
    "posterior distribution"
    postMean_bps_d100_n1000_slow = mcmc_bps_d100_n1000_slow.get_samples()['beta'].mean(0)
    "get samples from predictive distribution"
    postSamp_bps_d100_n1000_slow = mcmc_bps_d100_n1000_slow.get_samples()['beta']
    predSamp_bps_d100_n1000_slow = predictive_samples(postSamp_bps_d100_n1000_slow, data_d100_n1000)
    "SAVE TO CSV"
    postSamp_bps_d100_n1000_slow_df = pd.DataFrame(postSamp_bps_d100_n1000_slow.numpy())
    postSamp_bps_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/postSamp_bps_d100_n1000_slow_run" + str(i) + ".csv")

    "summary of predictions"
    predMean_bps_d100_n1000_slow ,predLower_bps_d100_n1000_slow, predUpper_bps_d100_n1000_slow = predictive_summary(predSamp_bps_d100_n1000_slow, 0.025)
    print("bps r2", r2_score(labels.squeeze(), predMean_bps_d100_n1000_slow))
    print("bps distance", torch.norm(postMean_bps_d100_n1000_slow - truePost.transpose(0,-1)))
    print("bps percentage", percentage_correct(predLower_bps_d100_n1000_slow,predUpper_bps_d100_n1000_slow,true_y_d100_n1000))
    "r2 score"
    r2scores_bps_d100_n1000_slow.append(r2_score(labels.squeeze(), predMean_bps_d100_n1000_slow))
    perCorrect_bps_d100_n1000_slow.append(percentage_correct(predLower_bps_d100_n1000_slow,predUpper_bps_d100_n1000_slow,true_y_d100_n1000))
    distances_bps_d100_n1000_slow.append(torch.norm(postMean_bps_d100_n1000_slow - truePost.transpose(0,-1)))

    "Convergences"
    #ksd_bps_d100_n1000_slow = kernel_Stein_Discrepancies(bps_d100_n1000_slow, postSamp_bps_d100_n1000_slow)
    #convergence_bps_d100_n1000_slow.append(ksd_bps_d100_n1000_slow)
    #######################################ZZ########################################################
    "DEFINITION OF SAMPLER"
    zz_d100_n1000_slow = ZZ(model, dimension=dim, Q = Target_sigma_inv, excess_rate = 1, ihpp_sampler = 'Exact')
    mcmc_zz_d100_n1000_slow = MCMC(zz_d100_n1000_slow, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_zz_d100_n1000_slow.run(data_d100_n1000)
    "posterior distribution"
    postMean_zz_d100_n1000_slow = mcmc_zz_d100_n1000_slow.get_samples()['beta'].mean(0)
    "get samples from predictive distribution"
    postSamp_zz_d100_n1000_slow = mcmc_zz_d100_n1000_slow.get_samples()['beta']
    predSamp_zz_d100_n1000_slow = predictive_samples(postSamp_zz_d100_n1000_slow, data_d100_n1000)
    "SAVE TO CSV"
    postSamp_zz_d100_n1000_slow_df = pd.DataFrame(postSamp_zz_d100_n1000_slow.numpy())
    postSamp_zz_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/postSamp_zz_d100_n1000_slow_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_zz_d100_n1000_slow ,predLower_zz_d100_n1000_slow, predUpper_zz_d100_n1000_slow = predictive_summary(predSamp_zz_d100_n1000_slow, 0.025)
    print("zz r2", r2_score(labels.squeeze(), predMean_zz_d100_n1000_slow))
    print("zz distance", torch.norm(postMean_zz_d100_n1000_slow - truePost.transpose(0,-1)))
    print("zz percentage", percentage_correct(predLower_zz_d100_n1000_slow,predUpper_zz_d100_n1000_slow,true_y_d100_n1000))
    "r2 score"
    r2scores_zz_d100_n1000_slow.append(r2_score(labels.squeeze(), predMean_zz_d100_n1000_slow))
    perCorrect_zz_d100_n1000_slow.append(percentage_correct(predLower_zz_d100_n1000_slow,predUpper_zz_d100_n1000_slow,true_y_d100_n1000))
    distances_zz_d100_n1000_slow.append(torch.norm(postMean_zz_d100_n1000_slow - truePost.transpose(0,-1)))

    "Convergences"
    #ksd_zz_d100_n1000_slow = kernel_Stein_Discrepancies(zz_d100_n1000_slow, postSamp_zz_d100_n1000_slow)
    #convergence_zz_d100_n1000_slow.append(ksd_zz_d100_n1000_slow)
    #######################################HMC########################################################
    "DEFINITION OF SAMPLER"
    hmc_d100_n1000_slow = NUTS(model)
    mcmc_hmc_d100_n1000_slow = MCMC(hmc_d100_n1000_slow, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_hmc_d100_n1000_slow.run(data_d100_n1000)
    "posterior distribution"
    postMean_hmc_d100_n1000_slow = mcmc_hmc_d100_n1000_slow.get_samples()['beta'].mean(0)
    "get samples from predictive distribution"
    postSamp_hmc_d100_n1000_slow = mcmc_hmc_d100_n1000_slow.get_samples()['beta']
    predSamp_hmc_d100_n1000_slow = predictive_samples(postSamp_hmc_d100_n1000_slow, data_d100_n1000)
    "SAVE TO CSV"
    postSamp_hmc_d100_n1000_slow_df = pd.DataFrame(postSamp_hmc_d100_n1000_slow.numpy())
    postSamp_hmc_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/postSamp_hmc_d100_n1000_slow_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_hmc_d100_n1000_slow ,predLower_hmc_d100_n1000_slow, predUpper_hmc_d100_n1000_slow = predictive_summary(predSamp_hmc_d100_n1000_slow, 0.025)
    print("hmc r2", r2_score(labels.squeeze(), predMean_hmc_d100_n1000_slow))
    print("hmc distance", torch.norm(postMean_hmc_d100_n1000_slow - truePost.transpose(0,-1)))
    print("hmc percentage", percentage_correct(predLower_hmc_d100_n1000_slow,predUpper_hmc_d100_n1000_slow,true_y_d100_n1000))
    "r2 score"
    r2scores_hmc_d100_n1000_slow.append(r2_score(labels.squeeze(), predMean_hmc_d100_n1000_slow))
    perCorrect_hmc_d100_n1000_slow.append(percentage_correct(predLower_hmc_d100_n1000_slow,predUpper_hmc_d100_n1000_slow,true_y_d100_n1000))
    distances_hmc_d100_n1000_slow.append(torch.norm(postMean_hmc_d100_n1000_slow - truePost.transpose(0,-1)))

    "Convergences"
    #ksd_hmc_d100_n1000_slow = kernel_Stein_Discrepancies(hmc_d100_n1000_slow, postSamp_hmc_d100_n1000_slow)
    #convergence_hmc_d100_n1000_slow.append(ksd_hmc_d100_n1000_slow)

"to pandas bk"
r2scores_bk_d100_n1000_slow_df = pd.DataFrame(r2scores_bk_d100_n1000_slow)
perCorrect_bk_d100_n1000_slow_df = pd.DataFrame(perCorrect_bk_d100_n1000_slow)
distances_bk_d100_n1000_slow_df = pd.DataFrame(distances_bk_d100_n1000_slow)
"to csv bps"
r2scores_bk_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/r2scores_bk_d100_n1000_slow.csv")
perCorrect_bk_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/perCorrect_bk_d100_n1000_slow.csv")
distances_bk_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/distances_bk_d100_n1000_slow.csv")

"to pandas bps"
r2scores_bps_d100_n1000_slow_df = pd.DataFrame(r2scores_bps_d100_n1000_slow)
perCorrect_bps_d100_n1000_slow_df = pd.DataFrame(perCorrect_bps_d100_n1000_slow)
distances_bps_d100_n1000_slow_df = pd.DataFrame(distances_bps_d100_n1000_slow)
"to csv bps"
r2scores_bps_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/r2scores_bps_d100_n1000_slow.csv")
perCorrect_bps_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/perCorrect_bps_d100_n1000_slow.csv")
distances_bps_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/distances_bps_d100_n1000_slow.csv")

"to pandas zz"
r2scores_zz_d100_n1000_slow_df = pd.DataFrame(r2scores_zz_d100_n1000_slow)
perCorrect_zz_d100_n1000_slow_df = pd.DataFrame(perCorrect_zz_d100_n1000_slow)
distances_zz_d100_n1000_slow_df = pd.DataFrame(distances_zz_d100_n1000_slow)
"to csv bps"
r2scores_zz_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/r2scores_zz_d100_n1000_slow.csv")
perCorrect_zz_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/perCorrect_zz_d100_n1000_slow.csv")
distances_zz_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/distances_zz_d100_n1000_slow.csv")

"to pandas hmc"
r2scores_hmc_d100_n1000_slow_df = pd.DataFrame(r2scores_hmc_d100_n1000_slow)
perCorrect_hmc_d100_n1000_slow_df = pd.DataFrame(perCorrect_hmc_d100_n1000_slow)
distances_hmc_d100_n1000_slow_df = pd.DataFrame(distances_hmc_d100_n1000_slow)
"to csv bps"
r2scores_hmc_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/r2scores_hmc_d100_n1000_slow.csv")
perCorrect_hmc_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/perCorrect_hmc_d100_n1000_slow.csv")
distances_hmc_d100_n1000_slow_df.to_csv(PATH + "/results/d100_n1000_slow/distances_hmc_d100_n1000_slow.csv")
