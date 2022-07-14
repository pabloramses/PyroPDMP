import numpy as np
from bps import bps
import matplotlib.pyplot as plt
Sigma = np.array([[1,-0.7],[-0.7,1]])
Mu = np.array([10,7])
Sigma_inv = np.linalg.inv(Sigma)
def mult_gaussian_gradient(x):
    grad = np.zeros(len(x))
    for i in range(len(x)):
        grad[i] = np.dot(x-Mu, Sigma_inv[i,:])
    return grad

def bound(t, i, x, v):
    a = np.dot(np.abs(Sigma_inv[0, :]), np.abs(x - Mu))
    b = np.sum(np.abs(Sigma_inv[0, :]))
    for i in range(1, len(x)):
        a = a + np.dot(np.abs(Sigma_inv[i, :]), np.abs(x - Mu))
        b = b + np.sum(np.abs(Sigma_inv[i, :]))
    return a + t*b

def inv_int_bound(t, x, v):
    a = np.dot(np.abs(Sigma_inv[0, :]), np.abs(x - Mu))
    b = np.sum(np.abs(Sigma_inv[0, :]))
    for i in range(1, len(x)):
        a = a + np.dot(np.abs(Sigma_inv[i, :]), np.abs(x - Mu))
        b = b + np.sum(np.abs(Sigma_inv[i, :]))
    return (-a + np.sqrt((a ** 2) + 2 * b * t)) / b

initial_pos = np.array([0.5,1])
sampler = bps(mult_gaussian_gradient, bound, initial_pos, niter = 100000, lr = 1, inv_int_bound= inv_int_bound)
sampler.sample()

plt.plot(sampler.pos[0], sampler.pos[1])