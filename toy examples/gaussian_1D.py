import numpy as np
from zzsampler import zigzag
from bps import bps
import matplotlib.pyplot as plt
Sigma = 1
def gaussian_gradient(x):
    return x/Sigma

def bound(t, i, x, v):
    return np.abs(x*v)/Sigma + t/Sigma

def inv_int_bound(t, x, v):
    a = np.abs(v*x)/Sigma
    b = 1/Sigma
    return (-a+np.sqrt((a**2)+2*b*t))/b

initial_pos = np.array([-1])
sampler = zigzag(gaussian_gradient, bound, initial_pos, niter = 1000, inv_int_bound= inv_int_bound)
sampler.sample()

bsamp = bps(gaussian_gradient, bound, initial_pos, niter = 1000, lr = 1, inv_int_bound=inv_int_bound)

bsamp.sample()

plt.hist(bsamp.pos[0], density = True)