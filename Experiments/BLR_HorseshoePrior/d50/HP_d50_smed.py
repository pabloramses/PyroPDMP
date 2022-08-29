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

# True Model
dim_mod = 50
total_dim = 2 * dim_mod + 2
sample_size = 1000
"Definition of the model"


def model(data):
    coefs_mean = torch.zeros(dim_mod)
    lambdas = pyro.sample('lambdas', dist.HalfCauchy(1).expand([dim_mod]))
    tau = pyro.sample('tau', dist.HalfCauchy(1))
    sigma = pyro.sample('sigma', dist.Gamma(1, 1))
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, (tau * lambdas) ** 2))
    y = pyro.sample('y', dist.Normal((coefs * data).sum(-1), sigma), obs=labels)
    return y


num_samples = 1000
warmup_steps = 1000
K = 1000

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

for i in range(5):
    mu_betas = torch.randint(1, 3, size=(1, dim_mod))[0] * 1.0
    coefs_100 = torch.normal(mean=mu_betas, std=torch.ones(dim_mod) * 0.2)
    sparse_coefs_100 = torch.randint(0, 2, (dim_mod,)) * 1.0
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
    sparsity_bk_d50_smed.append(sparsity_bk_d50_smed_i)
    print("bk sparsity", sparsity_bk_d50_smed_i)
    "get samples from predictive distribution"
    postSamp_bk_d50_smed = mcmc_bk_d50_smed.get_samples()['beta']
    predSamp_bk_d50_smed = predictive_samples(postSamp_bk_d50_smed, data_d50)
    "SAVE TO CSV"
    postSamp_bk_d50_smed_df = pd.DataFrame(postSamp_bk_d50_smed.numpy())
    postSamp_bk_d50_smed_df.to_csv(PATH + "/results/d50_smed/postSamp_bk_d50_smed_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_bk_d50_smed, predLower_bk_d50_smed, predUpper_bk_d50_smed = predictive_summary(predSamp_bk_d50_smed,
                                                                                               0.025)
    print("bk r2", r2_score(labels.squeeze(), predMean_bk_d50_smed))
    print("bk percentage", percentage_correct(predLower_bk_d50_smed, predUpper_bk_d50_smed, true_y_d50))

    "Scores"
    r2scores_bk_d50_smed.append(r2_score(labels.squeeze(), predMean_bk_d50_smed))
    perCorrect_bk_d50_smed.append(percentage_correct(predLower_bk_d50_smed, predUpper_bk_d50_smed, true_y_d50))

    "Convergences"
    ksd_bk_d50_smed = KSD_hp_2(bk_d50_smed, mcmc_bk_d50_smed.get_samples(),c=100000, beta=0.9, K=K)
    convergence_bk_d50_smed.append(ksd_bk_d50_smed)
    print("bk convergence", ksd_bk_d50_smed)

    #######################################BPS########################################################
    "DEFINITION OF SAMPLER"
    bps_d50_smed = BPS(model, dimension=total_dim, refresh_rate=10, ihpp_sampler='Corbella')
    mcmc_bps_d50_smed = MCMC(bps_d50_smed, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_bps_d50_smed.run(data_d50)
    "posterior distribution"
    postMean_bps_d50_smed = mcmc_bps_d50_smed.get_samples()['beta'].mean(0)
    sparsity_bps_d50_smed_i = sparsity(sparse_coefs_100, postMean_bps_d50_smed)
    sparsity_bps_d50_smed.append(sparsity_bps_d50_smed_i)
    print("bps sparsity", sparsity_bps_d50_smed_i)
    "get samples from predictive distribution"
    postSamp_bps_d50_smed = mcmc_bps_d50_smed.get_samples()['beta']
    predSamp_bps_d50_smed = predictive_samples(postSamp_bps_d50_smed, data_d50)
    "SAVE TO CSV"
    postSamp_bps_d50_smed_df = pd.DataFrame(postSamp_bps_d50_smed.numpy())
    postSamp_bps_d50_smed_df.to_csv(PATH + "/results/d50_smed/postSamp_bps_d50_smed_run" + str(i) + ".csv")

    "summary of predictions"
    predMean_bps_d50_smed, predLower_bps_d50_smed, predUpper_bps_d50_smed = predictive_summary(
        predSamp_bps_d50_smed, 0.025)
    print("bps r2", r2_score(labels.squeeze(), predMean_bps_d50_smed))
    print("bps percentage", percentage_correct(predLower_bps_d50_smed, predUpper_bps_d50_smed, true_y_d50))
    "r2 score"
    r2scores_bps_d50_smed.append(r2_score(labels.squeeze(), predMean_bps_d50_smed))
    perCorrect_bps_d50_smed.append(percentage_correct(predLower_bps_d50_smed, predUpper_bps_d50_smed, true_y_d50))

    "Convergences"
    ksd_bps_d50_smed = KSD_hp_2(bps_d50_smed, mcmc_bps_d50_smed.get_samples(),c=100000, beta=0.9, K=K)
    convergence_bps_d50_smed.append(ksd_bps_d50_smed)
    print("bps convergence", ksd_bps_d50_smed)
    """
    #######################################ZZ########################################################
    "DEFINITION OF SAMPLER"
    zz_d50_smed = ZZ(model, dimension=total_dim, excess_rate=0.1, ihpp_sampler='Corbella')
    mcmc_zz_d50_smed = MCMC(zz_d50_smed, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_zz_d50_smed.run(data_d50)
    "posterior distribution"
    postMean_zz_d50_smed = mcmc_zz_d50_smed.get_samples()['beta'].mean(0)
    sparsity_zz_d50_smed_i = sparsity(sparse_coefs_100, postMean_zz_d50_smed)
    sparsity_zz_d50_smed.append(sparsity_zz_d50_smed_i)
    print("zz sparsity", sparsity_zz_d50_smed_i)
    "get samples from predictive distribution"
    postSamp_zz_d50_smed = mcmc_zz_d50_smed.get_samples()['beta']
    predSamp_zz_d50_smed = predictive_samples(postSamp_zz_d50_smed, data_d50)
    "SAVE TO CSV"
    postSamp_zz_d50_smed_df = pd.DataFrame(postSamp_zz_d50_smed.numpy())
    postSamp_zz_d50_smed_df.to_csv(PATH + "/results/d50_smed/postSamp_zz_d50_smed_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_zz_d50_smed, predLower_zz_d50_smed, predUpper_zz_d50_smed = predictive_summary(predSamp_zz_d50_smed,
                                                                                               0.025)
    print("zz r2", r2_score(labels.squeeze(), predMean_zz_d50_smed))
    print("zz percentage", percentage_correct(predLower_zz_d50_smed, predUpper_zz_d50_smed, true_y_d50))
    "r2 score"
    r2scores_zz_d50_smed.append(r2_score(labels.squeeze(), predMean_zz_d50_smed))
    perCorrect_zz_d50_smed.append(percentage_correct(predLower_zz_d50_smed, predUpper_zz_d50_smed, true_y_d50))

    "Convergences"
    ksd_zz_d50_smed = KSD_hp_2(zz_d50_smed, mcmc_zz_d50_smed.get_samples(),c=100000, beta=0.9, K=K)
    convergence_zz_d50_smed.append(ksd_zz_d50_smed)
    print("zz convergence", ksd_zz_d50_smed)
    """
    #######################################HMC########################################################
    "DEFINITION OF SAMPLER"
    hmc_d50_smed = NUTS(model)
    mcmc_hmc_d50_smed = MCMC(hmc_d50_smed, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_hmc_d50_smed.run(data_d50)
    "posterior distribution"
    postMean_hmc_d50_smed = mcmc_hmc_d50_smed.get_samples()['beta'].mean(0)
    sparsity_hmc_d50_smed_i = sparsity(sparse_coefs_100, postMean_hmc_d50_smed)
    sparsity_hmc_d50_smed.append(sparsity_hmc_d50_smed_i)
    print("hmc sparsity", sparsity_hmc_d50_smed_i)
    "get samples from predictive distribution"
    postSamp_hmc_d50_smed = mcmc_hmc_d50_smed.get_samples()['beta']
    predSamp_hmc_d50_smed = predictive_samples(postSamp_hmc_d50_smed, data_d50)
    "SAVE TO CSV"
    postSamp_hmc_d50_smed_df = pd.DataFrame(postSamp_hmc_d50_smed.numpy())
    postSamp_hmc_d50_smed_df.to_csv(PATH + "/results/d50_smed/postSamp_hmc_d50_smed_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_hmc_d50_smed, predLower_hmc_d50_smed, predUpper_hmc_d50_smed = predictive_summary(
        predSamp_hmc_d50_smed, 0.025)
    print("hmc r2", r2_score(labels.squeeze(), predMean_hmc_d50_smed))
    print("hmc percentage", percentage_correct(predLower_hmc_d50_smed, predUpper_hmc_d50_smed, true_y_d50))
    "r2 score"
    r2scores_hmc_d50_smed.append(r2_score(labels.squeeze(), predMean_hmc_d50_smed))
    perCorrect_hmc_d50_smed.append(percentage_correct(predLower_hmc_d50_smed, predUpper_hmc_d50_smed, true_y_d50))

    "Convergences"
    ksd_hmc_d50_smed = KSD_hp_2(hmc_d50_smed, mcmc_hmc_d50_smed.get_samples(),c=100000, beta=0.9, K=K)
    convergence_hmc_d50_smed.append(ksd_hmc_d50_smed)
    print("hmc convergence", ksd_hmc_d50_smed)
"to pandas bk"
r2scores_bk_d50_smed_df = pd.DataFrame(r2scores_bk_d50_smed)
perCorrect_bk_d50_smed_df = pd.DataFrame(perCorrect_bk_d50_smed)
convergence_bk_d50_smed_df = pd.DataFrame(convergence_bk_d50_smed)
sparsity_bk_d50_smed_df = pd.DataFrame(sparsity_bk_d50_smed)

"to csv bk"
r2scores_bk_d50_smed_df.to_csv(PATH + "/results/d50_smed/r2scores_bk_d50_smed.csv")
perCorrect_bk_d50_smed_df.to_csv(PATH + "/results/d50_smed/perCorrect_bk_d50_smed.csv")
convergence_bk_d50_smed_df.to_csv(PATH + "/results/d50_smed/convergence_bk_d50_smed.csv")
sparsity_bk_d50_smed_df.to_csv(PATH + "/results/d50_slow/sparsity_bk_d50_smed.csv")

"to pandas bps"
r2scores_bps_d50_smed_df = pd.DataFrame(r2scores_bps_d50_smed)
perCorrect_bps_d50_smed_df = pd.DataFrame(perCorrect_bps_d50_smed)
convergence_bps_d50_smed_df = pd.DataFrame(convergence_bps_d50_smed)
sparsity_bps_d50_smed_df = pd.DataFrame(sparsity_bps_d50_smed)
"to csv bps"
r2scores_bps_d50_smed_df.to_csv(PATH + "/results/d50_smed/r2scores_bps_d50_smed.csv")
perCorrect_bps_d50_smed_df.to_csv(PATH + "/results/d50_smed/perCorrect_bps_d50_smed.csv")
convergence_bps_d50_smed_df.to_csv(PATH + "/results/d50_smed/convergence_bps_d50_smed.csv")
sparsity_bps_d50_smed_df.to_csv(PATH + "/results/d50_slow/sparsity_bps_d50_smed.csv")
"""
"to pandas zz"
r2scores_zz_d50_smed_df = pd.DataFrame(r2scores_zz_d50_smed)
perCorrect_zz_d50_smed_df = pd.DataFrame(perCorrect_zz_d50_smed)
convergence_zz_d50_smed_df = pd.DataFrame(convergence_zz_d50_smed)
sparsity_zz_d50_smed_df = pd.DataFrame(sparsity_zz_d50_smed)
"to csv bps"
r2scores_zz_d50_smed_df.to_csv(PATH + "/results/d50_smed/r2scores_zz_d50_smed.csv")
perCorrect_zz_d50_smed_df.to_csv(PATH + "/results/d50_smed/perCorrect_zz_d50_smed.csv")
convergence_zz_d50_smed_df.to_csv(PATH + "/results/d50_smed/convergence_zz_d50_smed.csv")
sparsity_zz_d50_smed_df.to_csv(PATH + "/results/d50_slow/sparsity_zz_d50_smed.csv")
"""

"to pandas hmc"
r2scores_hmc_d50_smed_df = pd.DataFrame(r2scores_hmc_d50_smed)
perCorrect_hmc_d50_smed_df = pd.DataFrame(perCorrect_hmc_d50_smed)
convergence_hmc_d50_smed_df = pd.DataFrame(convergence_hmc_d50_smed)
sparsity_hmc_d50_smed_df = pd.DataFrame(sparsity_hmc_d50_smed)
"to csv bps"
r2scores_hmc_d50_smed_df.to_csv(PATH + "/results/d50_smed/r2scores_hmc_d50_smed.csv")
perCorrect_hmc_d50_smed_df.to_csv(PATH + "/results/d50_smed/perCorrect_hmc_d50_smed.csv")
convergence_hmc_d50_smed_df.to_csv(PATH + "/results/d50_smed/convergence_hmc_d50_smed.csv")
sparsity_hmc_d50_smed_df.to_csv(PATH + "/results/d50_slow/sparsity_hmc_d50_smed.csv")