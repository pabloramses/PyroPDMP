import numpy as np


def normal_density(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-(1/(2*sigma)) * (x - mu)**2)

def betas_gradient(betas, lambdas, tau, sigma, X_design=None, y=None):
    mu_post = (1/sigma) * np.dot(np.linalg.inv((1/(tau**2)) * np.diag(1/lambdas**2) + (1/sigma) * np.dot(X_design.transpose(), X_design)), np.dot(X_design.transpose(), y))
    Sigma_post_inv = (1/(tau**2)) * np.diag(1/lambdas**2) + (1/sigma) * np.dot(X_design.transpose(), X_design)
    return np.dot(betas-mu_post, Sigma_post_inv)

def metropolis_conditionals(betas, lambdas, tau):
    for j in range(len(lambdas)):
        proposal = np.abs(np.random.standard_cauchy())
        if np.random.rand() < normal_density(betas[j], 0, (lambdas[j]**2) * tau):
            lambdas[j] = proposal
    proposal_2 = np.abs(np.random.standard_cauchy())
    #calculation of the acceptance ratio
    numerator = 1
    denominator = 1
    for j in range(len(lambdas)):
        numerator = numerator * normal_density(betas[j], 0, (lambdas[j]**2) * proposal)
        denominator = denominator * normal_density(betas[j], 0, (lambdas[j]**2) * tau)

    if np.random.rand() < (numerator / denominator):
        tau = proposal

    return lambdas, tau

