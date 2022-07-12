import numpy as np
from zzsampler import zigzag
import matplotlib.pyplot as plt
Sigma = np.array([[1,0],[0,1]])
Mu = np.array([1,1])
Sigma_inv = np.linalg.inv(Sigma)
def mult_gaussian_gradient(x):
    grad = np.zeros(len(x))
    for i in range(len(x)):
        grad[i] = np.dot(x-Mu, Sigma_inv[i,:])
    return grad

def bound(t, i, x, v):
    a = np.dot(np.abs(Sigma_inv[i,:]),np.abs(x-Mu))
    b = np.sum(Sigma_inv[i,:])
    return a + t*b

def inv_int_bound(t, x, v):
    evTimes = np.array([])
    for i in range(len(t)):
        a = np.dot(np.abs(Sigma_inv[i, :]), np.abs(x - Mu))
        b = np.sum(np.abs(Sigma_inv[i, :]))
        evTimes = np.append(evTimes, (-a+np.sqrt((a**2)+2*b*t[i]))/b)
        print(evTimes)
    return evTimes

initial_pos = np.array([-1,7])
sampler = zigzag(mult_gaussian_gradient, bound, initial_pos, niter = 1000, inv_int_bound= inv_int_bound)
sampler.sample()

plt.plot(sampler.pos[0], sampler.pos[1])