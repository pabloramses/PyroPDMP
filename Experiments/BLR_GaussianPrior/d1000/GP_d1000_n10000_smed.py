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

#True Model
def model(data):
    coefs_mean = torch.zeros(dim)
    coefs_var = torch.ones(dim)
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, coefs_var))
    y = pyro.sample('y', dist.Normal((coefs * data).sum(-1), sigma), obs=labels)
    return y



sample_size = 10000
dim = 1000

num_samples = 10000
warmup_steps = 10000



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

for i in range(10):
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
    bk_d1000_n10000_smed = Boomerang(model, Sigma=Sigma_ref, hessian_bound = hessian_bound, refresh_rate = 1, ihpp_sampler = 'Exact')
    mcmc_bk_d1000_n10000_smed = MCMC(bk_d1000_n10000_smed, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_bk_d1000_n10000_smed.run(data_d1000_n10000)
    "posterior distribution"
    postMean_bk_d1000_n10000_smed = mcmc_bk_d1000_n10000_smed.get_samples()['beta'].mean(0)
    "get samples from predictive distribution"
    postSamp_bk_d1000_n10000_smed = mcmc_bk_d1000_n10000_smed.get_samples()['beta']
    print("bk distance", torch.norm(postMean_bk_d1000_n10000_smed - truePost.transpose(0,-1)))
    predSamp_bk_d1000_n10000_smed = predictive_samples(postSamp_bk_d1000_n10000_smed, data_d1000_n10000)
    "SAVE TO CSV"
    postSamp_bk_d1000_n10000_smed_df = pd.DataFrame(postSamp_bk_d1000_n10000_smed.numpy())
    postSamp_bk_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/postSamp_bk_d1000_n10000_smed_run"+str(i)+".csv")
    "summary of predictions"
    predMean_bk_d1000_n10000_smed ,predLower_bk_d1000_n10000_smed, predUpper_bk_d1000_n10000_smed = predictive_summary(predSamp_bk_d1000_n10000_smed, 0.025)
    print("bk r2", r2_score(labels.squeeze(), predMean_bk_d1000_n10000_smed))
    print("bk percentage", percentage_correct(predLower_bk_d1000_n10000_smed,predUpper_bk_d1000_n10000_smed,true_y_d1000_n10000))

    "Scores"
    r2scores_bk_d1000_n10000_smed.append(r2_score(labels.squeeze(), predMean_bk_d1000_n10000_smed))
    perCorrect_bk_d1000_n10000_smed.append(percentage_correct(predLower_bk_d1000_n10000_smed,predUpper_bk_d1000_n10000_smed,true_y_d1000_n10000))
    distances_bk_d1000_n10000_smed.append(torch.norm(postMean_bk_d1000_n10000_smed - truePost.transpose(0,-1)))

    "Convergences"
    #k_bk_d1000_n10000_smed = kernel_Stein_Discrepancies(bk_d1000_n10000_smed, postSamp_bk_d1000_n10000_smed)
    #convergence_bk_d1000_n10000_smed.append(k_bk_d1000_n10000_smed)

    #######################################BPS########################################################
    "DEFINITION OF SAMPLER"
    bps_d1000_n10000_smed = BPS(model, dimension=dim, hessian_bound = hessian_bound, refresh_rate = 10, ihpp_sampler = 'Exact')
    mcmc_bps_d1000_n10000_smed = MCMC(bps_d1000_n10000_smed, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_bps_d1000_n10000_smed.run(data_d1000_n10000)
    "posterior distribution"
    postMean_bps_d1000_n10000_smed = mcmc_bps_d1000_n10000_smed.get_samples()['beta'].mean(0)
    "get samples from predictive distribution"
    postSamp_bps_d1000_n10000_smed = mcmc_bps_d1000_n10000_smed.get_samples()['beta']
    print("bps distance", torch.norm(postMean_bps_d1000_n10000_smed - truePost.transpose(0,-1)))
    predSamp_bps_d1000_n10000_smed = predictive_samples(postSamp_bps_d1000_n10000_smed, data_d1000_n10000)
    "SAVE TO CSV"
    postSamp_bps_d1000_n10000_smed_df = pd.DataFrame(postSamp_bps_d1000_n10000_smed.numpy())
    postSamp_bps_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/postSamp_bps_d1000_n10000_smed_run" + str(i) + ".csv")

    "summary of predictions"
    predMean_bps_d1000_n10000_smed ,predLower_bps_d1000_n10000_smed, predUpper_bps_d1000_n10000_smed = predictive_summary(predSamp_bps_d1000_n10000_smed, 0.025)
    print("bps r2", r2_score(labels.squeeze(), predMean_bps_d1000_n10000_smed))
    print("bps percentage", percentage_correct(predLower_bps_d1000_n10000_smed,predUpper_bps_d1000_n10000_smed,true_y_d1000_n10000))
    "r2 score"
    r2scores_bps_d1000_n10000_smed.append(r2_score(labels.squeeze(), predMean_bps_d1000_n10000_smed))
    perCorrect_bps_d1000_n10000_smed.append(percentage_correct(predLower_bps_d1000_n10000_smed,predUpper_bps_d1000_n10000_smed,true_y_d1000_n10000))
    distances_bps_d1000_n10000_smed.append(torch.norm(postMean_bps_d1000_n10000_smed - truePost.transpose(0,-1)))

    "Convergences"
    #ksd_bps_d1000_n10000_smed = kernel_Stein_Discrepancies(bps_d1000_n10000_smed, postSamp_bps_d1000_n10000_smed)
    #convergence_bps_d1000_n10000_smed.append(ksd_bps_d1000_n10000_smed)
    #######################################ZZ########################################################
    "DEFINITION OF SAMPLER"
    zz_d1000_n10000_smed = ZZ(model, dimension=dim, Q = Target_sigma_inv, excess_rate = 1, ihpp_sampler = 'Exact')
    mcmc_zz_d1000_n10000_smed = MCMC(zz_d1000_n10000_smed, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_zz_d1000_n10000_smed.run(data_d1000_n10000)
    "posterior distribution"
    postMean_zz_d1000_n10000_smed = mcmc_zz_d1000_n10000_smed.get_samples()['beta'].mean(0)
    "get samples from predictive distribution"
    postSamp_zz_d1000_n10000_smed = mcmc_zz_d1000_n10000_smed.get_samples()['beta']
    print("zz distance", torch.norm(postMean_zz_d1000_n10000_smed - truePost.transpose(0,-1)))
    predSamp_zz_d1000_n10000_smed = predictive_samples(postSamp_zz_d1000_n10000_smed, data_d1000_n10000)
    "SAVE TO CSV"
    postSamp_zz_d1000_n10000_smed_df = pd.DataFrame(postSamp_zz_d1000_n10000_smed.numpy())
    postSamp_zz_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/postSamp_zz_d1000_n10000_smed_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_zz_d1000_n10000_smed ,predLower_zz_d1000_n10000_smed, predUpper_zz_d1000_n10000_smed = predictive_summary(predSamp_zz_d1000_n10000_smed, 0.025)
    print("zz r2", r2_score(labels.squeeze(), predMean_zz_d1000_n10000_smed))
    print("zz percentage", percentage_correct(predLower_zz_d1000_n10000_smed,predUpper_zz_d1000_n10000_smed,true_y_d1000_n10000))
    "r2 score"
    r2scores_zz_d1000_n10000_smed.append(r2_score(labels.squeeze(), predMean_zz_d1000_n10000_smed))
    perCorrect_zz_d1000_n10000_smed.append(percentage_correct(predLower_zz_d1000_n10000_smed,predUpper_zz_d1000_n10000_smed,true_y_d1000_n10000))
    distances_zz_d1000_n10000_smed.append(torch.norm(postMean_zz_d1000_n10000_smed - truePost.transpose(0,-1)))

    "Convergences"
    #ksd_zz_d1000_n10000_smed = kernel_Stein_Discrepancies(zz_d1000_n10000_smed, postSamp_zz_d1000_n10000_smed)
    #convergence_zz_d1000_n10000_smed.append(ksd_zz_d1000_n10000_smed)
    #######################################HMC########################################################
    "DEFINITION OF SAMPLER"
    hmc_d1000_n10000_smed = NUTS(model)
    mcmc_hmc_d1000_n10000_smed = MCMC(hmc_d1000_n10000_smed, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_hmc_d1000_n10000_smed.run(data_d1000_n10000)
    "posterior distribution"
    postMean_hmc_d1000_n10000_smed = mcmc_hmc_d1000_n10000_smed.get_samples()['beta'].mean(0)
    "get samples from predictive distribution"
    postSamp_hmc_d1000_n10000_smed = mcmc_hmc_d1000_n10000_smed.get_samples()['beta']
    print("hmc distance", torch.norm(postMean_hmc_d1000_n10000_smed - truePost.transpose(0,-1)))
    predSamp_hmc_d1000_n10000_smed = predictive_samples(postSamp_hmc_d1000_n10000_smed, data_d1000_n10000)
    "SAVE TO CSV"
    postSamp_hmc_d1000_n10000_smed_df = pd.DataFrame(postSamp_hmc_d1000_n10000_smed.numpy())
    postSamp_hmc_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/postSamp_hmc_d1000_n10000_smed_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_hmc_d1000_n10000_smed ,predLower_hmc_d1000_n10000_smed, predUpper_hmc_d1000_n10000_smed = predictive_summary(predSamp_hmc_d1000_n10000_smed, 0.025)
    print("hmc r2", r2_score(labels.squeeze(), predMean_hmc_d1000_n10000_smed))
    print("hmc percentage", percentage_correct(predLower_hmc_d1000_n10000_smed,predUpper_hmc_d1000_n10000_smed,true_y_d1000_n10000))
    "r2 score"
    r2scores_hmc_d1000_n10000_smed.append(r2_score(labels.squeeze(), predMean_hmc_d1000_n10000_smed))
    perCorrect_hmc_d1000_n10000_smed.append(percentage_correct(predLower_hmc_d1000_n10000_smed,predUpper_hmc_d1000_n10000_smed,true_y_d1000_n10000))
    distances_hmc_d1000_n10000_smed.append(torch.norm(postMean_hmc_d1000_n10000_smed - truePost.transpose(0,-1)))

    "Convergences"
    #ksd_hmc_d1000_n10000_smed = kernel_Stein_Discrepancies(hmc_d1000_n10000_smed, postSamp_hmc_d1000_n10000_smed)
    #convergence_hmc_d1000_n10000_smed.append(ksd_hmc_d1000_n10000_smed)

"to pandas bk"
r2scores_bk_d1000_n10000_smed_df = pd.DataFrame(r2scores_bk_d1000_n10000_smed)
perCorrect_bk_d1000_n10000_smed_df = pd.DataFrame(perCorrect_bk_d1000_n10000_smed)
distances_bk_d1000_n10000_smed_df = pd.DataFrame(distances_bk_d1000_n10000_smed)
"to csv bps"
r2scores_bk_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/r2scores_bk_d1000_n10000_smed.csv")
perCorrect_bk_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/perCorrect_bk_d1000_n10000_smed.csv")
distances_bk_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/distances_bk_d1000_n10000_smed.csv")

"to pandas bps"
r2scores_bps_d1000_n10000_smed_df = pd.DataFrame(r2scores_bps_d1000_n10000_smed)
perCorrect_bps_d1000_n10000_smed_df = pd.DataFrame(perCorrect_bps_d1000_n10000_smed)
distances_bps_d1000_n10000_smed_df = pd.DataFrame(distances_bps_d1000_n10000_smed)
"to csv bps"
r2scores_bps_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/r2scores_bps_d1000_n10000_smed.csv")
perCorrect_bps_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/perCorrect_bps_d1000_n10000_smed.csv")
distances_bps_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/distances_bps_d1000_n10000_smed.csv")

"to pandas zz"
r2scores_zz_d1000_n10000_smed_df = pd.DataFrame(r2scores_zz_d1000_n10000_smed)
perCorrect_zz_d1000_n10000_smed_df = pd.DataFrame(perCorrect_zz_d1000_n10000_smed)
distances_zz_d1000_n10000_smed_df = pd.DataFrame(distances_zz_d1000_n10000_smed)
"to csv bps"
r2scores_zz_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/r2scores_zz_d1000_n10000_smed.csv")
perCorrect_zz_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/perCorrect_zz_d1000_n10000_smed.csv")
distances_zz_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/distances_zz_d1000_n10000_smed.csv")

"to pandas hmc"
r2scores_hmc_d1000_n10000_smed_df = pd.DataFrame(r2scores_hmc_d1000_n10000_smed)
perCorrect_hmc_d1000_n10000_smed_df = pd.DataFrame(perCorrect_hmc_d1000_n10000_smed)
distances_hmc_d1000_n10000_smed_df = pd.DataFrame(distances_hmc_d1000_n10000_smed)
"to csv bps"
r2scores_hmc_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/r2scores_hmc_d1000_n10000_smed.csv")
perCorrect_hmc_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/perCorrect_hmc_d1000_n10000_smed.csv")
distances_hmc_d1000_n10000_smed_df.to_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_GaussianPrior/results/d1000_n10000_smed/distances_hmc_d1000_n10000_smed.csv")