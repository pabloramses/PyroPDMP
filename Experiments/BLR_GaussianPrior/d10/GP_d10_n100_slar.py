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
num_samples = 5000
warmup_steps = 5000
r2scores_bk_d10_n100_slar = []
perCorrect_bk_d10_n100_slar = []
distances_bk_d10_n100_slar = []
convergence_bk_d10_n100_slar = []

r2scores_bps_d10_n100_slar = []
perCorrect_bps_d10_n100_slar = []
distances_bps_d10_n100_slar = []
convergence_bps_d10_n100_slar = []

r2scores_hmc_d10_n100_slar = []
perCorrect_hmc_d10_n100_slar = []
distances_hmc_d10_n100_slar = []
convergence_hmc_d10_n100_slar = []

r2scores_zz_d10_n100_slar = []
perCorrect_zz_d10_n100_slar = []
distances_zz_d10_n100_slar = []
convergence_zz_d10_n100_slar = []

for i in range(10):
    mu_betas = torch.randint(-3,3, size=(1,10))[0]*1.0
    true_coefs_10 = torch.normal(mean=mu_betas, std=torch.ones(10))

    "10 SAMPLES"
    data_d10_n100 = torch.randn(100, 10)
    true_y_d10_n100 = torch.matmul(data_d10_n100, true_coefs_10)
    dim = 10

    "lar noise SNR = 5"
    SNR = 5
    sigma_lar = true_y_d10_n100.var(0) / SNR
    y_d10_n100_slar = true_y_d10_n100 + torch.normal(0, sigma_lar, size = (1, 100))


    "Tune model"
    errors = y_d10_n100_slar.transpose(0,-1) - torch.matmul(data_d10_n100,torch.matmul(torch.matmul(torch.inverse(torch.matmul(data_d10_n100.transpose(0,-1), data_d10_n100)), data_d10_n100.transpose(0,-1)), y_d10_n100_slar.transpose(0,-1)))
    sigma = torch.sqrt(torch.mean(errors**2)) #MLE estimation of noise
    labels = y_d10_n100_slar
    truePost = torch.matmul(torch.inverse(torch.eye(dim) + (1/sigma**2) * torch.matmul(data_d10_n100.transpose(0,-1),data_d10_n100)) , (1/sigma**2) * torch.matmul(data_d10_n100.transpose(0,-1), labels.transpose(0,-1)))
    #################################BOOMERANG##################################
    "BOUND ON HESSIAN"
    Target_sigma_inv = torch.eye(dim) + (1/sigma_lar**2) * torch.matmul(data_d10_n100.transpose(0,-1), data_d10_n100)
    hessian_bound = torch.linalg.matrix_norm(Target_sigma_inv).item()
    "REFERENCE MEASURE TUNING"
    Sigma_ref = torch.eye(dim)
    "DEFINITION OF SAMPLER"
    bk_d10_n100_slar = Boomerang(model, Sigma=Sigma_ref, hessian_bound = hessian_bound, refresh_rate = 1, ihpp_sampler = 'Exact')
    mcmc_bk_d10_n100_slar = MCMC(bk_d10_n100_slar, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_bk_d10_n100_slar.run(data_d10_n100)
    "posterior distribution"
    postMean_bk_d10_n100_slar = mcmc_bk_d10_n100_slar.get_samples()['beta'].mean(0)
    "get samples from predictive distribution"
    postSamp_bk_d10_n100_slar = mcmc_bk_d10_n100_slar.get_samples()['beta']
    predSamp_bk_d10_n100_slar = predictive_samples(postSamp_bk_d10_n100_slar, data_d10_n100)
    "SAVE TO CSV"
    postSamp_bk_d10_n100_slar_df = pd.DataFrame(postSamp_bk_d10_n100_slar.numpy())
    postSamp_bk_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/postSamp_bk_d10_n100_slar_run"+str(i)+".csv")
    "summary of predictions"
    predMean_bk_d10_n100_slar ,predlarer_bk_d10_n100_slar, predUpper_bk_d10_n100_slar = predictive_summary(predSamp_bk_d10_n100_slar, 0.025)


    "Scores"
    r2scores_bk_d10_n100_slar.append(r2_score(labels.squeeze(), predMean_bk_d10_n100_slar))
    perCorrect_bk_d10_n100_slar.append(percentage_correct(predlarer_bk_d10_n100_slar,predUpper_bk_d10_n100_slar,true_y_d10_n100))
    distances_bk_d10_n100_slar.append(torch.norm(postMean_bk_d10_n100_slar - truePost.transpose(0,-1)))

    "Convergences"
    #k_bk_d10_n100_slar = kernel_Stein_Discrepancies(bk_d10_n100_slar, postSamp_bk_d10_n100_slar[-1000:,:])
    #convergence_bk_d10_n100_slar.append(k_bk_d10_n100_slar)

    #######################################BPS########################################################
    "DEFINITION OF SAMPLER"
    bps_d10_n100_slar = BPS(model, dimension=dim, hessian_bound = hessian_bound, refresh_rate = 1, ihpp_sampler = 'Exact')
    mcmc_bps_d10_n100_slar = MCMC(bps_d10_n100_slar, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_bps_d10_n100_slar.run(data_d10_n100)
    "posterior distribution"
    postMean_bps_d10_n100_slar = mcmc_bps_d10_n100_slar.get_samples()['beta'].mean(0)
    "get samples from predictive distribution"
    postSamp_bps_d10_n100_slar = mcmc_bps_d10_n100_slar.get_samples()['beta']
    predSamp_bps_d10_n100_slar = predictive_samples(postSamp_bps_d10_n100_slar, data_d10_n100)
    "SAVE TO CSV"
    postSamp_bps_d10_n100_slar_df = pd.DataFrame(postSamp_bps_d10_n100_slar.numpy())
    postSamp_bps_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/postSamp_bps_d10_n100_slar_run" + str(i) + ".csv")

    "summary of predictions"
    predMean_bps_d10_n100_slar ,predlarer_bps_d10_n100_slar, predUpper_bps_d10_n100_slar = predictive_summary(predSamp_bps_d10_n100_slar, 0.025)

    "r2 score"
    r2scores_bps_d10_n100_slar.append(r2_score(labels.squeeze(), predMean_bps_d10_n100_slar))
    perCorrect_bps_d10_n100_slar.append(percentage_correct(predlarer_bps_d10_n100_slar,predUpper_bps_d10_n100_slar,true_y_d10_n100))
    distances_bps_d10_n100_slar.append(torch.norm(postMean_bps_d10_n100_slar - truePost.transpose(0,-1)))

    "Convergences"
    #ksd_bps_d10_n100_slar = kernel_Stein_Discrepancies(bps_d10_n100_slar, postSamp_bps_d10_n100_slar[-1000:,:])
    #convergence_bps_d10_n100_slar.append(ksd_bps_d10_n100_slar)
    #######################################ZZ########################################################
    "DEFINITION OF SAMPLER"
    zz_d10_n100_slar = ZZ(model, dimension=dim, Q = Target_sigma_inv, excess_rate = 0.25, ihpp_sampler = 'Exact')
    mcmc_zz_d10_n100_slar = MCMC(zz_d10_n100_slar, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_zz_d10_n100_slar.run(data_d10_n100)
    "posterior distribution"
    postMean_zz_d10_n100_slar = mcmc_zz_d10_n100_slar.get_samples()['beta'].mean(0)
    "get samples from predictive distribution"
    postSamp_zz_d10_n100_slar = mcmc_zz_d10_n100_slar.get_samples()['beta']
    predSamp_zz_d10_n100_slar = predictive_samples(postSamp_zz_d10_n100_slar, data_d10_n100)
    "SAVE TO CSV"
    postSamp_zz_d10_n100_slar_df = pd.DataFrame(postSamp_zz_d10_n100_slar.numpy())
    postSamp_zz_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/postSamp_zz_d10_n100_slar_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_zz_d10_n100_slar ,predlarer_zz_d10_n100_slar, predUpper_zz_d10_n100_slar = predictive_summary(predSamp_zz_d10_n100_slar, 0.025)

    "r2 score"
    r2scores_zz_d10_n100_slar.append(r2_score(labels.squeeze(), predMean_zz_d10_n100_slar))
    perCorrect_zz_d10_n100_slar.append(percentage_correct(predlarer_zz_d10_n100_slar,predUpper_zz_d10_n100_slar,true_y_d10_n100))
    distances_zz_d10_n100_slar.append(torch.norm(postMean_zz_d10_n100_slar - truePost.transpose(0,-1)))

    "Convergences"
    #ksd_zz_d10_n100_slar = kernel_Stein_Discrepancies(zz_d10_n100_slar, postSamp_zz_d10_n100_slar[-1000:,:])
    #convergence_zz_d10_n100_slar.append(ksd_zz_d10_n100_slar)
    #######################################HMC########################################################
    "DEFINITION OF SAMPLER"
    hmc_d10_n100_slar = NUTS(model)
    mcmc_hmc_d10_n100_slar = MCMC(hmc_d10_n100_slar, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_hmc_d10_n100_slar.run(data_d10_n100)
    "posterior distribution"
    postMean_hmc_d10_n100_slar = mcmc_hmc_d10_n100_slar.get_samples()['beta'].mean(0)
    "get samples from predictive distribution"
    postSamp_hmc_d10_n100_slar = mcmc_hmc_d10_n100_slar.get_samples()['beta']
    predSamp_hmc_d10_n100_slar = predictive_samples(postSamp_hmc_d10_n100_slar, data_d10_n100)
    "SAVE TO CSV"
    postSamp_hmc_d10_n100_slar_df = pd.DataFrame(postSamp_hmc_d10_n100_slar.numpy())
    postSamp_hmc_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/postSamp_hmc_d10_n100_slar_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_hmc_d10_n100_slar ,predlarer_hmc_d10_n100_slar, predUpper_hmc_d10_n100_slar = predictive_summary(predSamp_hmc_d10_n100_slar, 0.025)

    "r2 score"
    r2scores_hmc_d10_n100_slar.append(r2_score(labels.squeeze(), predMean_hmc_d10_n100_slar))
    perCorrect_hmc_d10_n100_slar.append(percentage_correct(predlarer_hmc_d10_n100_slar,predUpper_hmc_d10_n100_slar,true_y_d10_n100))
    distances_hmc_d10_n100_slar.append(torch.norm(postMean_hmc_d10_n100_slar - truePost.transpose(0,-1)))

    "Convergences"
    #ksd_hmc_d10_n100_slar = kernel_Stein_Discrepancies(hmc_d10_n100_slar, postSamp_hmc_d10_n100_slar[-1000:,:])
    #convergence_hmc_d10_n100_slar.append(ksd_hmc_d10_n100_slar[-1000:,:])

"to pandas bk"
r2scores_bk_d10_n100_slar_df = pd.DataFrame(r2scores_bk_d10_n100_slar)
perCorrect_bk_d10_n100_slar_df = pd.DataFrame(perCorrect_bk_d10_n100_slar)
distances_bk_d10_n100_slar_df = pd.DataFrame(distances_bk_d10_n100_slar)
convergence_bk_d10_n100_slar_df = pd.DataFrame(convergence_bk_d10_n100_slar)

"to csv bk"
r2scores_bk_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/r2scores_bk_d10_n100_slar.csv")
perCorrect_bk_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/perCorrect_bk_d10_n100_slar.csv")
distances_bk_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/distances_bk_d10_n100_slar.csv")
convergence_bk_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/convergence_bk_d10_n100_slar.csv")

"to pandas bps"
r2scores_bps_d10_n100_slar_df = pd.DataFrame(r2scores_bps_d10_n100_slar)
perCorrect_bps_d10_n100_slar_df = pd.DataFrame(perCorrect_bps_d10_n100_slar)
distances_bps_d10_n100_slar_df = pd.DataFrame(distances_bps_d10_n100_slar)
convergence_bps_d10_n100_slar_df = pd.DataFrame(convergence_bps_d10_n100_slar)
"to csv bps"
r2scores_bps_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/r2scores_bps_d10_n100_slar.csv")
perCorrect_bps_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/perCorrect_bps_d10_n100_slar.csv")
distances_bps_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/distances_bps_d10_n100_slar.csv")
convergence_bps_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/convergence_bps_d10_n100_slar.csv")
"to pandas zz"
r2scores_zz_d10_n100_slar_df = pd.DataFrame(r2scores_zz_d10_n100_slar)
perCorrect_zz_d10_n100_slar_df = pd.DataFrame(perCorrect_zz_d10_n100_slar)
distances_zz_d10_n100_slar_df = pd.DataFrame(distances_zz_d10_n100_slar)
convergence_zz_d10_n100_slar_df = pd.DataFrame(convergence_zz_d10_n100_slar)
"to csv bps"
r2scores_zz_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/r2scores_zz_d10_n100_slar.csv")
perCorrect_zz_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/perCorrect_zz_d10_n100_slar.csv")
distances_zz_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/distances_zz_d10_n100_slar.csv")
convergence_zz_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/convergence_zz_d10_n100_slar.csv")
"to pandas hmc"
r2scores_hmc_d10_n100_slar_df = pd.DataFrame(r2scores_hmc_d10_n100_slar)
perCorrect_hmc_d10_n100_slar_df = pd.DataFrame(perCorrect_hmc_d10_n100_slar)
distances_hmc_d10_n100_slar_df = pd.DataFrame(distances_hmc_d10_n100_slar)
convergence_hmc_d10_n100_slar_df = pd.DataFrame(convergence_hmc_d10_n100_slar)
"to csv bps"
r2scores_hmc_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/r2scores_hmc_d10_n100_slar.csv")
perCorrect_hmc_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/perCorrect_hmc_d10_n100_slar.csv")
distances_hmc_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/distances_hmc_d10_n100_slar.csv")
convergence_hmc_d10_n100_slar_df.to_csv(PATH + "/results/d10_n100_slar/convergence_hmc_d10_n100_slar.csv")