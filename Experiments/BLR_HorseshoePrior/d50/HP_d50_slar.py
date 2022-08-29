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

r2scores_bk_d50_slar = []
sparsity_bk_d50_slar = []
perCorrect_bk_d50_slar = []
convergence_bk_d50_slar = []

r2scores_bps_d50_slar = []
sparsity_bps_d50_slar = []
perCorrect_bps_d50_slar = []
convergence_bps_d50_slar = []

r2scores_hmc_d50_slar = []
sparsity_hmc_d50_slar = []
perCorrect_hmc_d50_slar = []
convergence_hmc_d50_slar = []

r2scores_zz_d50_slar = []
sparsity_zz_d50_slar = []
perCorrect_zz_d50_slar = []
convergence_zz_d50_slar = []

for i in range(5):
    mu_betas = torch.randint(1, 3, size=(1, dim_mod))[0] * 1.0
    coefs_100 = torch.normal(mean=mu_betas, std=torch.ones(dim_mod) * 0.2)
    sparse_coefs_100 = torch.bernoulli(torch.ones(dim_mod)*0.2)
    true_coefs_100 = torch.normal(mean=mu_betas, std=torch.ones(dim_mod)) * sparse_coefs_100

    "10 SAMPLES"
    data_d50 = torch.randn(sample_size, dim_mod)
    true_y_d50 = torch.matmul(data_d50, true_coefs_100)

    "med noise SNR = 50"
    SNR = 7.5
    sigma_med = true_y_d50.var(0) / SNR
    y_d50_slar = true_y_d50 + torch.normal(0, sigma_med, size=(1, sample_size))

    "Tune model"
    labels = y_d50_slar

    #################################BOOMERANG##################################
    "REFERENCE MEASURE TUNING"
    Sigma_ref = torch.eye(total_dim)
    "DEFINITION OF SAMPLER"
    bk_d50_slar = Boomerang(model, Sigma=Sigma_ref, refresh_rate=10, ihpp_sampler='Corbella')
    mcmc_bk_d50_slar = MCMC(bk_d50_slar, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_bk_d50_slar.run(data_d50)
    "posterior distribution"
    postMean_bk_d50_slar = mcmc_bk_d50_slar.get_samples()['beta'].mean(0)
    sparsity_bk_d50_slar_i = sparsity(sparse_coefs_100, postMean_bk_d50_slar)
    sparsity_bk_d50_slar.append(sparsity_bk_d50_slar_i)
    print("bk sparsity", sparsity_bk_d50_slar)
    "get samples from predictive distribution"
    postSamp_bk_d50_slar = mcmc_bk_d50_slar.get_samples()['beta']
    predSamp_bk_d50_slar = predictive_samples(postSamp_bk_d50_slar, data_d50)
    "SAVE TO CSV"
    postSamp_bk_d50_slar_df = pd.DataFrame(postSamp_bk_d50_slar.numpy())
    postSamp_bk_d50_slar_df.to_csv(PATH + "/results/d50_slar/postSamp_bk_d50_slar_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_bk_d50_slar, predLower_bk_d50_slar, predUpper_bk_d50_slar = predictive_summary(predSamp_bk_d50_slar,
                                                                                               0.025)
    print("bk r2", r2_score(labels.squeeze(), predMean_bk_d50_slar))
    print("bk percentage", percentage_correct(predLower_bk_d50_slar, predUpper_bk_d50_slar, true_y_d50))

    "Scores"
    r2scores_bk_d50_slar.append(r2_score(labels.squeeze(), predMean_bk_d50_slar))
    perCorrect_bk_d50_slar.append(percentage_correct(predLower_bk_d50_slar, predUpper_bk_d50_slar, true_y_d50))

    "Convergences"
    ksd_bk_d50_slar = KSD_hp_2(bk_d50_slar, mcmc_bk_d50_slar.get_samples(),c=100000, beta=0.9, K=K)
    convergence_bk_d50_slar.append(ksd_bk_d50_slar)
    print("bk convergence", ksd_bk_d50_slar)

    #######################################BPS########################################################
    "DEFINITION OF SAMPLER"
    bps_d50_slar = BPS(model, dimension=total_dim, refresh_rate=10, ihpp_sampler='Corbella')
    mcmc_bps_d50_slar = MCMC(bps_d50_slar, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_bps_d50_slar.run(data_d50)
    "posterior distribution"
    postMean_bps_d50_slar = mcmc_bps_d50_slar.get_samples()['beta'].mean(0)
    sparsity_bps_d50_slar_i = sparsity(sparse_coefs_100, postMean_bps_d50_slar)
    sparsity_bps_d50_slar.append(sparsity_bps_d50_slar_i)
    print("bps sparsity", sparsity_bps_d50_slar)
    "get samples from predictive distribution"
    postSamp_bps_d50_slar = mcmc_bps_d50_slar.get_samples()['beta']
    predSamp_bps_d50_slar = predictive_samples(postSamp_bps_d50_slar, data_d50)
    "SAVE TO CSV"
    postSamp_bps_d50_slar_df = pd.DataFrame(postSamp_bps_d50_slar.numpy())
    postSamp_bps_d50_slar_df.to_csv(PATH + "/results/d50_slar/postSamp_bps_d50_slar_run" + str(i) + ".csv")

    "summary of predictions"
    predMean_bps_d50_slar, predLower_bps_d50_slar, predUpper_bps_d50_slar = predictive_summary(
        predSamp_bps_d50_slar, 0.025)
    print("bps r2", r2_score(labels.squeeze(), predMean_bps_d50_slar))
    print("bps percentage", percentage_correct(predLower_bps_d50_slar, predUpper_bps_d50_slar, true_y_d50))
    "r2 score"
    r2scores_bps_d50_slar.append(r2_score(labels.squeeze(), predMean_bps_d50_slar))
    perCorrect_bps_d50_slar.append(percentage_correct(predLower_bps_d50_slar, predUpper_bps_d50_slar, true_y_d50))

    "Convergences"
    ksd_bps_d50_slar = KSD_hp_2(bps_d50_slar, mcmc_bps_d50_slar.get_samples(),c=100000, beta=0.9, K=K)
    convergence_bps_d50_slar.append(ksd_bps_d50_slar)
    print("bps convergence", ksd_bps_d50_slar)
    """
    #######################################ZZ########################################################
    "DEFINITION OF SAMPLER"
    zz_d50_slar = ZZ(model, dimension=total_dim, excess_rate=0.1, ihpp_sampler='Corbella')
    mcmc_zz_d50_slar = MCMC(zz_d50_slar, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_zz_d50_slar.run(data_d50)
    "posterior distribution"
    postMean_zz_d50_slar = mcmc_zz_d50_slar.get_samples()['beta'].mean(0)
    sparsity_zz_d50_slar_i = sparsity(sparse_coefs_100, postMean_zz_d50_slar)
    sparsity_zz_d50_slar.append(sparsity_zz_d50_slar_i)
    print("zz sparsity", sparsity_zz_d50_slar)
    "get samples from predictive distribution"
    postSamp_zz_d50_slar = mcmc_zz_d50_slar.get_samples()['beta']
    predSamp_zz_d50_slar = predictive_samples(postSamp_zz_d50_slar, data_d50)
    "SAVE TO CSV"
    postSamp_zz_d50_slar_df = pd.DataFrame(postSamp_zz_d50_slar.numpy())
    postSamp_zz_d50_slar_df.to_csv(PATH + "/results/d50_slar/postSamp_zz_d50_slar_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_zz_d50_slar, predLower_zz_d50_slar, predUpper_zz_d50_slar = predictive_summary(predSamp_zz_d50_slar,
                                                                                               0.025)
    print("zz r2", r2_score(labels.squeeze(), predMean_zz_d50_slar))
    print("zz percentage", percentage_correct(predLower_zz_d50_slar, predUpper_zz_d50_slar, true_y_d50))
    "r2 score"
    r2scores_zz_d50_slar.append(r2_score(labels.squeeze(), predMean_zz_d50_slar))
    perCorrect_zz_d50_slar.append(percentage_correct(predLower_zz_d50_slar, predUpper_zz_d50_slar, true_y_d50))

    "Convergences"
    ksd_zz_d50_slar = KSD_hp_2(zz_d50_slar, mcmc_zz_d50_slar.get_samples(),c=100000, beta=0.9, K=K)
    convergence_zz_d50_slar.append(ksd_zz_d50_slar)
    print("zz convergence", ksd_zz_d50_slar)
    """
    #######################################HMC########################################################
    "DEFINITION OF SAMPLER"
    hmc_d50_slar = NUTS(model)
    mcmc_hmc_d50_slar = MCMC(hmc_d50_slar, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_hmc_d50_slar.run(data_d50)
    "posterior distribution"
    postMean_hmc_d50_slar = mcmc_hmc_d50_slar.get_samples()['beta'].mean(0)
    sparsity_hmc_d50_slar_i = sparsity(sparse_coefs_100, postMean_hmc_d50_slar)
    sparsity_hmc_d50_slar.append(sparsity_hmc_d50_slar_i)
    print("hmc sparsity", sparsity_hmc_d50_slar_i)
    "get samples from predictive distribution"
    postSamp_hmc_d50_slar = mcmc_hmc_d50_slar.get_samples()['beta']
    predSamp_hmc_d50_slar = predictive_samples(postSamp_hmc_d50_slar, data_d50)
    "SAVE TO CSV"
    postSamp_hmc_d50_slar_df = pd.DataFrame(postSamp_hmc_d50_slar.numpy())
    postSamp_hmc_d50_slar_df.to_csv(PATH + "/results/d50_slar/postSamp_hmc_d50_slar_run" + str(i) + ".csv")
    "summary of predictions"
    predMean_hmc_d50_slar, predLower_hmc_d50_slar, predUpper_hmc_d50_slar = predictive_summary(
        predSamp_hmc_d50_slar, 0.025)
    print("hmc r2", r2_score(labels.squeeze(), predMean_hmc_d50_slar))
    print("hmc percentage", percentage_correct(predLower_hmc_d50_slar, predUpper_hmc_d50_slar, true_y_d50))
    "r2 score"
    r2scores_hmc_d50_slar.append(r2_score(labels.squeeze(), predMean_hmc_d50_slar))
    perCorrect_hmc_d50_slar.append(percentage_correct(predLower_hmc_d50_slar, predUpper_hmc_d50_slar, true_y_d50))

    "Convergences"
    ksd_hmc_d50_slar = KSD_hp_2(hmc_d50_slar, mcmc_hmc_d50_slar.get_samples(),c=100000, beta=0.9, K=K)
    convergence_hmc_d50_slar.append(ksd_hmc_d50_slar)
    print("hmc convergence", ksd_hmc_d50_slar)
"to pandas bk"
r2scores_bk_d50_slar_df = pd.DataFrame(r2scores_bk_d50_slar)
perCorrect_bk_d50_slar_df = pd.DataFrame(perCorrect_bk_d50_slar)
convergence_bk_d50_slar_df = pd.DataFrame(convergence_bk_d50_slar)
sparsity_bk_d50_slar_df = pd.DataFrame(sparsity_bk_d50_slar)

"to csv bk"
r2scores_bk_d50_slar_df.to_csv(PATH + "/results/d50_slar/r2scores_bk_d50_slar.csv")
perCorrect_bk_d50_slar_df.to_csv(PATH + "/results/d50_slar/perCorrect_bk_d50_slar.csv")
convergence_bk_d50_slar_df.to_csv(PATH + "/results/d50_slar/convergence_bk_d50_slar.csv")
sparsity_bk_d50_slar_df.to_csv(PATH + "/results/d50_slow/sparsity_bk_d50_slar.csv")

"to pandas bps"
r2scores_bps_d50_slar_df = pd.DataFrame(r2scores_bps_d50_slar)
perCorrect_bps_d50_slar_df = pd.DataFrame(perCorrect_bps_d50_slar)
convergence_bps_d50_slar_df = pd.DataFrame(convergence_bps_d50_slar)
sparsity_bps_d50_slar_df = pd.DataFrame(sparsity_bps_d50_slar)
"to csv bps"
r2scores_bps_d50_slar_df.to_csv(PATH + "/results/d50_slar/r2scores_bps_d50_slar.csv")
perCorrect_bps_d50_slar_df.to_csv(PATH + "/results/d50_slar/perCorrect_bps_d50_slar.csv")
convergence_bps_d50_slar_df.to_csv(PATH + "/results/d50_slar/convergence_bps_d50_slar.csv")
sparsity_bps_d50_slar_df.to_csv(PATH + "/results/d50_slow/sparsity_bps_d50_slar.csv")
"""
"to pandas zz"
r2scores_zz_d50_slar_df = pd.DataFrame(r2scores_zz_d50_slar)
perCorrect_zz_d50_slar_df = pd.DataFrame(perCorrect_zz_d50_slar)
convergence_zz_d50_slar_df = pd.DataFrame(convergence_zz_d50_slar)
sparsity_zz_d50_slar_df = pd.DataFrame(sparsity_zz_d50_slar)
"to csv bps"
r2scores_zz_d50_slar_df.to_csv(PATH + "/results/d50_slar/r2scores_zz_d50_slar.csv")
perCorrect_zz_d50_slar_df.to_csv(PATH + "/results/d50_slar/perCorrect_zz_d50_slar.csv")
convergence_zz_d50_slar_df.to_csv(PATH + "/results/d50_slar/convergence_zz_d50_slar.csv")
sparsity_zz_d50_slar_df.to_csv(PATH + "/results/d50_slow/sparsity_zz_d50_slar.csv")
"""
"to pandas hmc"
r2scores_hmc_d50_slar_df = pd.DataFrame(r2scores_hmc_d50_slar)
perCorrect_hmc_d50_slar_df = pd.DataFrame(perCorrect_hmc_d50_slar)
convergence_hmc_d50_slar_df = pd.DataFrame(convergence_hmc_d50_slar)
sparsity_hmc_d50_slar_df = pd.DataFrame(sparsity_hmc_d50_slar)
"to csv bps"
r2scores_hmc_d50_slar_df.to_csv(PATH + "/results/d50_slar/r2scores_hmc_d50_slar.csv")
perCorrect_hmc_d50_slar_df.to_csv(PATH + "/results/d50_slar/perCorrect_hmc_d50_slar.csv")
convergence_hmc_d50_slar_df.to_csv(PATH + "/results/d50_slar/convergence_hmc_d50_slar.csv")
sparsity_hmc_d50_slar_df.to_csv(PATH + "/results/d50_slow/sparsity_hmc_d50_slar.csv")