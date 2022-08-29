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
K=1000

r2scores_bk_d50_slow = []
sparsity_bk_d50_slow = []
perCorrect_bk_d50_slow = []
convergence_bk_d50_slow = []

r2scores_bps_d50_slow = []
sparsity_bps_d50_slow = []
perCorrect_bps_d50_slow = []
convergence_bps_d50_slow = []

r2scores_hmc_d50_slow = []
sparsity_hmc_d50_slow = []
perCorrect_hmc_d50_slow = []
convergence_hmc_d50_slow = []

r2scores_zz_d50_slow = []
sparsity_zz_d50_slow = []
perCorrect_zz_d50_slow = []
convergence_zz_d50_slow = []

for i in range(5):
    mu_betas = torch.randint(1, 3, size=(1, dim_mod))[0] * 1.0
    coefs_100 = torch.normal(mean=mu_betas, std=torch.ones(dim_mod) * 0.8)
    sparse_coefs_100 = torch.bernoulli(torch.ones(dim_mod)*0.8)
    true_coefs_100 = torch.normal(mean=mu_betas, std=torch.ones(dim_mod)) * sparse_coefs_100

    "10 SAMPLES"
    data_d50 = torch.randn(sample_size, dim_mod)
    true_y_d50 = torch.matmul(data_d50, true_coefs_100)

    "med noise SNR = 50"
    SNR = 15
    sigma_med = true_y_d50.var(0) / SNR
    y_d50_slow = true_y_d50 + torch.normal(0, sigma_med, size=(1, sample_size))

    "Tune model"
    labels = y_d50_slow

    #################################BOOMERANG##################################
    "REFERENCE MEASURE TUNING"
    Sigma_ref = torch.eye(total_dim)
    "DEFINITION OF SAMPLER"
    bk_d50_slow = Boomerang(model, Sigma=Sigma_ref, refresh_rate=10, ihpp_sampler='Corbella')
    mcmc_bk_d50_slow = MCMC(bk_d50_slow, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_bk_d50_slow.run(data_d50)
    "posterior distribution"
    postMean_bk_d50_slow = mcmc_bk_d50_slow.get_samples()['beta'].mean(0)
    sparsity_bk_d50_slow_i = sparsity(sparse_coefs_100, postMean_bk_d50_slow)
    sparsity_bk_d50_slow.append(sparsity_bk_d50_slow_i)
    print("bk sparsity", sparsity_bk_d50_slow_i)
    "get samples from predictive distribution"
    postSamp_bk_d50_slow = mcmc_bk_d50_slow.get_samples()['beta']
    predSamp_bk_d50_slow = predictive_samples(postSamp_bk_d50_slow, data_d50)
    "SAVE TO CSV"
    postSamp_bk_d50_slow_df = pd.DataFrame(postSamp_bk_d50_slow.numpy())
    postSamp_bk_d50_slow_df.to_csv(PATH + "/results/d50_slow/postSamp_bk_d50_slow_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_bk_d50_slow, predLower_bk_d50_slow, predUpper_bk_d50_slow = predictive_summary(predSamp_bk_d50_slow,
                                                                                               0.025)
    print("bk r2", r2_score(labels.squeeze(), predMean_bk_d50_slow))
    print("bk percentage", percentage_correct(predLower_bk_d50_slow, predUpper_bk_d50_slow, true_y_d50))

    "Scores"
    r2scores_bk_d50_slow.append(r2_score(labels.squeeze(), predMean_bk_d50_slow))
    perCorrect_bk_d50_slow.append(percentage_correct(predLower_bk_d50_slow, predUpper_bk_d50_slow, true_y_d50))

    "Convergences"
    ksd_bk_d50_slow = KSD_hp_2(bk_d50_slow, mcmc_bk_d50_slow.get_samples(),c=100000, beta=0.9, K=K)
    convergence_bk_d50_slow.append(ksd_bk_d50_slow)
    print("bk convergence", ksd_bk_d50_slow)

    #######################################BPS########################################################
    "DEFINITION OF SAMPLER"
    bps_d50_slow = BPS(model, dimension=total_dim, refresh_rate=10, ihpp_sampler='Corbella')
    mcmc_bps_d50_slow = MCMC(bps_d50_slow, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_bps_d50_slow.run(data_d50)
    "posterior distribution"
    postMean_bps_d50_slow = mcmc_bps_d50_slow.get_samples()['beta'].mean(0)
    sparsity_bps_d50_slow_i = sparsity(sparse_coefs_100, postMean_bps_d50_slow)
    sparsity_bps_d50_slow.append(sparsity_bps_d50_slow_i)
    print("bps sparsity", sparsity_bps_d50_slow_i)
    "get samples from predictive distribution"
    postSamp_bps_d50_slow = mcmc_bps_d50_slow.get_samples()['beta']
    predSamp_bps_d50_slow = predictive_samples(postSamp_bps_d50_slow, data_d50)
    "SAVE TO CSV"
    postSamp_bps_d50_slow_df = pd.DataFrame(postSamp_bps_d50_slow.numpy())
    postSamp_bps_d50_slow_df.to_csv(PATH + "/results/d50_slow/postSamp_bps_d50_slow_run" + str(i) + ".csv")

    "summary of predictions"
    predMean_bps_d50_slow, predLower_bps_d50_slow, predUpper_bps_d50_slow = predictive_summary(
        predSamp_bps_d50_slow, 0.025)
    print("bps r2", r2_score(labels.squeeze(), predMean_bps_d50_slow))
    print("bps percentage", percentage_correct(predLower_bps_d50_slow, predUpper_bps_d50_slow, true_y_d50))
    "r2 score"
    r2scores_bps_d50_slow.append(r2_score(labels.squeeze(), predMean_bps_d50_slow))
    perCorrect_bps_d50_slow.append(percentage_correct(predLower_bps_d50_slow, predUpper_bps_d50_slow, true_y_d50))

    "Convergences"
    ksd_bps_d50_slow = KSD_hp_2(bps_d50_slow, mcmc_bps_d50_slow.get_samples(),c=100000, beta=0.9, K=K)
    convergence_bps_d50_slow.append(ksd_bps_d50_slow)
    print("bps convergence", ksd_bps_d50_slow)
    """
    #######################################ZZ########################################################
    "DEFINITION OF SAMPLER"
    zz_d50_slow = ZZ(model, dimension=total_dim, excess_rate=0.1, ihpp_sampler='Corbella')
    mcmc_zz_d50_slow = MCMC(zz_d50_slow, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_zz_d50_slow.run(data_d50)
    "posterior distribution"
    postMean_zz_d50_slow = mcmc_zz_d50_slow.get_samples()['beta'].mean(0)
    sparsity_zz_d50_slow_i = sparsity(sparse_coefs_100, postMean_zz_d50_slow)
    sparsity_zz_d50_slow.append(sparsity_zz_d50_slow_i)
    print("zz sparsity", sparsity_zz_d50_slow_i)
    "get samples from predictive distribution"
    postSamp_zz_d50_slow = mcmc_zz_d50_slow.get_samples()['beta']
    predSamp_zz_d50_slow = predictive_samples(postSamp_zz_d50_slow, data_d50)
    "SAVE TO CSV"
    postSamp_zz_d50_slow_df = pd.DataFrame(postSamp_zz_d50_slow.numpy())
    postSamp_zz_d50_slow_df.to_csv(PATH + "/results/d50_slow/postSamp_zz_d50_slow_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_zz_d50_slow, predLower_zz_d50_slow, predUpper_zz_d50_slow = predictive_summary(predSamp_zz_d50_slow,
                                                                                               0.025)
    print("zz r2", r2_score(labels.squeeze(), predMean_zz_d50_slow))
    print("zz percentage", percentage_correct(predLower_zz_d50_slow, predUpper_zz_d50_slow, true_y_d50))
    "r2 score"
    r2scores_zz_d50_slow.append(r2_score(labels.squeeze(), predMean_zz_d50_slow))
    perCorrect_zz_d50_slow.append(percentage_correct(predLower_zz_d50_slow, predUpper_zz_d50_slow, true_y_d50))

    "Convergences"
    ksd_zz_d50_slow = KSD_hp_2(zz_d50_slow, mcmc_zz_d50_slow.get_samples(),c=100000, beta=0.9, K=K)
    convergence_zz_d50_slow.append(ksd_zz_d50_slow)
    print("zz convergence", ksd_zz_d50_slow)
    """
    #######################################HMC########################################################
    "DEFINITION OF SAMPLER"
    hmc_d50_slow = NUTS(model)
    mcmc_hmc_d50_slow = MCMC(hmc_d50_slow, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_hmc_d50_slow.run(data_d50)
    "posterior distribution"
    postMean_hmc_d50_slow = mcmc_hmc_d50_slow.get_samples()['beta'].mean(0)
    sparsity_hmc_d50_slow_i = sparsity(sparse_coefs_100, postMean_hmc_d50_slow)
    sparsity_hmc_d50_slow.append(sparsity_hmc_d50_slow_i)
    print("hmc sparsity", sparsity_hmc_d50_slow_i)
    "get samples from predictive distribution"
    postSamp_hmc_d50_slow = mcmc_hmc_d50_slow.get_samples()['beta']
    predSamp_hmc_d50_slow = predictive_samples(postSamp_hmc_d50_slow, data_d50)
    "SAVE TO CSV"
    postSamp_hmc_d50_slow_df = pd.DataFrame(postSamp_hmc_d50_slow.numpy())
    postSamp_hmc_d50_slow_df.to_csv(PATH + "/results/d50_slow/postSamp_hmc_d50_slow_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_hmc_d50_slow, predLower_hmc_d50_slow, predUpper_hmc_d50_slow = predictive_summary(
        predSamp_hmc_d50_slow, 0.025)
    print("hmc r2", r2_score(labels.squeeze(), predMean_hmc_d50_slow))
    print("hmc percentage", percentage_correct(predLower_hmc_d50_slow, predUpper_hmc_d50_slow, true_y_d50))
    "r2 score"
    r2scores_hmc_d50_slow.append(r2_score(labels.squeeze(), predMean_hmc_d50_slow))
    perCorrect_hmc_d50_slow.append(percentage_correct(predLower_hmc_d50_slow, predUpper_hmc_d50_slow, true_y_d50))

    "Convergences"
    ksd_hmc_d50_slow = KSD_hp_2(hmc_d50_slow, mcmc_hmc_d50_slow.get_samples(),c=100000, beta=0.9, K=K)
    convergence_hmc_d50_slow.append(ksd_hmc_d50_slow)
    print("hmc convergence", ksd_hmc_d50_slow)
"to pandas bk"
r2scores_bk_d50_slow_df = pd.DataFrame(r2scores_bk_d50_slow)
perCorrect_bk_d50_slow_df = pd.DataFrame(perCorrect_bk_d50_slow)
convergence_bk_d50_slow_df = pd.DataFrame(convergence_bk_d50_slow)
sparsity_bk_d50_slow_df = pd.DataFrame(sparsity_bk_d50_slow)

"to csv bk"
r2scores_bk_d50_slow_df.to_csv(PATH + "/results/d50_slow/r2scores_bk_d50_slow.csv")
perCorrect_bk_d50_slow_df.to_csv(PATH + "/results/d50_slow/perCorrect_bk_d50_slow.csv")
convergence_bk_d50_slow_df.to_csv(PATH + "/results/d50_slow/convergence_bk_d50_slow.csv")
sparsity_bk_d50_slow_df.to_csv(PATH + "/results/d50_slow/sparsity_bk_d50_slow.csv")

"to pandas bps"
r2scores_bps_d50_slow_df = pd.DataFrame(r2scores_bps_d50_slow)
perCorrect_bps_d50_slow_df = pd.DataFrame(perCorrect_bps_d50_slow)
convergence_bps_d50_slow_df = pd.DataFrame(convergence_bps_d50_slow)
sparsity_bps_d50_slow_df = pd.DataFrame(sparsity_bps_d50_slow)
"to csv bps"
r2scores_bps_d50_slow_df.to_csv(PATH + "/results/d50_slow/r2scores_bps_d50_slow.csv")
perCorrect_bps_d50_slow_df.to_csv(PATH + "/results/d50_slow/perCorrect_bps_d50_slow.csv")
convergence_bps_d50_slow_df.to_csv(PATH + "/results/d50_slow/convergence_bps_d50_slow.csv")
sparsity_bps_d50_slow_df.to_csv(PATH + "/results/d50_slow/sparsity_bps_d50_slow.csv")
"""
"to pandas zz"
r2scores_zz_d50_slow_df = pd.DataFrame(r2scores_zz_d50_slow)
perCorrect_zz_d50_slow_df = pd.DataFrame(perCorrect_zz_d50_slow)
convergence_zz_d50_slow_df = pd.DataFrame(convergence_zz_d50_slow)
sparsity_zz_d50_slow_df = pd.DataFrame(sparsity_zz_d50_slow)
"to csv bps"
r2scores_zz_d50_slow_df.to_csv(PATH + "/results/d50_slow/r2scores_zz_d50_slow.csv")
perCorrect_zz_d50_slow_df.to_csv(PATH + "/results/d50_slow/perCorrect_zz_d50_slow.csv")
convergence_zz_d50_slow_df.to_csv(PATH + "/results/d50_slow/convergence_zz_d50_slow.csv")
sparsity_zz_d50_slow_df.to_csv(PATH + "/results/d50_slow/sparsity_zz_d50_slow.csv")
"""
"to pandas hmc"
r2scores_hmc_d50_slow_df = pd.DataFrame(r2scores_hmc_d50_slow)
perCorrect_hmc_d50_slow_df = pd.DataFrame(perCorrect_hmc_d50_slow)
convergence_hmc_d50_slow_df = pd.DataFrame(convergence_hmc_d50_slow)
sparsity_hmc_d50_slow_df = pd.DataFrame(sparsity_hmc_d50_slow)
"to csv bps"
r2scores_hmc_d50_slow_df.to_csv(PATH + "/results/d50_slow/r2scores_hmc_d50_slow.csv")
perCorrect_hmc_d50_slow_df.to_csv(PATH + "/results/d50_slow/perCorrect_hmc_d50_slow.csv")
convergence_hmc_d50_slow_df.to_csv(PATH + "/results/d50_slow/convergence_hmc_d50_slow.csv")
sparsity_hmc_d50_slow_df.to_csv(PATH + "/results/d50_slow/sparsity_hmc_d50_slow.csv")