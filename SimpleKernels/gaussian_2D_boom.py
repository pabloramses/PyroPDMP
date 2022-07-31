import numpy as np
import matplotlib.pyplot as plt
from boomerang import boomerang

Sigma = np.array([[3,0],[0,3]])
Mu = np.array([3,4])
Sigma_inv = np.linalg.inv(Sigma)
def boomerang_gradient(x):
    grad = np.zeros(len(x))
    for i in range(len(x)):
        grad[i] =  np.dot(x-Mu, Sigma_inv[i,:]) #- np.dot(x-mu_ref, sigma_ref_inv[i,:])
    return grad

def bound(t, i, x, v):
    M = np.linalg.norm(Sigma_inv) + np.linalg.norm(sigma_ref_inv)
    m = np.linalg.norm(boomerang_gradient(x-mu_ref))
    a = (np.dot(v, boomerang_gradient(x-mu_ref))>0) * (np.dot(v, boomerang_gradient(x-mu_ref)))
    b = M * (np.linalg.norm(x-mu_ref)**2 + np.linalg.norm(v)**2) + m * np.sqrt((np.linalg.norm(x-mu_ref)**2 + np.linalg.norm(v)**2))
    return a + t*b

def inv_int_bound(t, x, v):
    M = np.linalg.norm(Sigma_inv) + np.linalg.norm(sigma_ref_inv)
    m = np.linalg.norm(boomerang_gradient(x-mu_ref))
    a = (np.dot(v, boomerang_gradient(x-mu_ref))>0) * (np.dot(v, boomerang_gradient(x-mu_ref)))
    b = M * (np.linalg.norm(x-mu_ref)**2 + np.linalg.norm(v)**2) + m * np.sqrt((np.linalg.norm(x-mu_ref)**2 + np.linalg.norm(v)**2))
    return (-a + np.sqrt((a ** 2) + 2 * b * t)) / b

initial_pos = np.array([3,4])
mu_ref = np.zeros(2)
sigma_ref = Sigma
sigma_ref_inv = Sigma_inv
sampler = boomerang(sigma_ref, mu_ref, boomerang_gradient, bound, initial_pos, niter = 10000, lr = 0.1, inv_int_bound= inv_int_bound)
sampler.sample()

plt.plot(sampler.pos[0], sampler.pos[1])
np.cov(sampler.pos[0], sampler.pos[1])
np.mean(sampler.pos[0])
np.mean(sampler.pos[1])