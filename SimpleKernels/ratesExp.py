import numpy as np
import matplotlib.pyplot as plt

Sigma = np.array([[3,0],[0,3]])
Mu = np.array([3,4])
Sigma_inv = np.linalg.inv(Sigma)

mu_ref = np.zeros(2)
sigma_ref = Sigma
sigma_ref_inv = Sigma_inv

def gradient(x):
    grad = np.zeros(len(x))
    #for i in range(len(x)):
    grad[0] =  x[0]**2
    grad[1] = x[1]**3 + np.sin(x[1])
    return grad

def lam(t):
    res = []
    for i in range(len(t)):
        a1 = - np.array([1,2]) * np.sin(t[i]) + np.array([0.5, -0.7])*np.cos(t[i])
        a2 = gradient(np.array([1,2]) * np.cos(t[i]) + np.array([0.5, -0.7])*np.sin(t[i]))
        res = np.append(res, (np.dot(a1, a2)>0)*np.dot(a1, a2))
    return res

t = np.linspace(0, 20, 100000) 

plt.plot(t, lam(t))