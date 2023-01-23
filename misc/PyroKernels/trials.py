import matplotlib.pyplot as plt
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC
from Pyro_Boomerang import Boomerang


num_samples = 1
warm_up = 1
true_coefs = torch.tensor([1., 2.])
data = torch.randn(2000, 2)
dim = 2
labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()
true_coefs = torch.tensor([1., 2.])
data = torch.randn(2000, 2)
dim = 2
labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()


def model(data):
    coefs_mean = torch.zeros(dim)
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(2)))
    y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
    return y

Sigma_ref = torch.eye(dim)
Sigma_ref_inv = Sigma_ref

boomerang_kernel = Boomerang(model,Sigma=Sigma_ref, refresh_rate=1.0, ihpp_sampler='Corbella')
mcmc = MCMC(boomerang_kernel, num_samples=num_samples, warmup_steps=warm_up)
mcmc.run(data)
skel_points = mcmc.get_samples()['beta']
print(mcmc.get_samples()[0])


def EllipticDynamics(t, y0, w0):
    # simulate dy/dt = w, dw/dt = -y
    y_new = y0 * np.cos(t) + w0 * np.sin(t)
    w_new = -y0 * np.sin(t) + w0 * np.cos(t)
    return (y_new, w_new)

def dict_of_tensors_to_numpy(z):
    if len(list(z.keys())) == 1:
        key_of_z = list(z.keys())[0]
        z_numpy = z[key_of_z].numpy()
    else:
        key_of_z = list(z.keys())
        z_numpy = z[key_of_z[0]][0].numpy()
        for j in range(1, len(key_of_z)):
            # convert to Numpy array
            z_numpy_j = z[key_of_z[j]][0].numpy()
            z_numpy = np.append(z_numpy, z_numpy_j)
        for t in range(1, z[key_of_z[0]].shape[1]):
            z_numpy_t = z[key_of_z[0]][t].numpy()
            for j in range(1, len(key_of_z)):
                # convert to Numpy array
                z_numpy_j = z[key_of_z[j]][t].numpy()
                z_numpy_t = np.append(z_numpy_t, z_numpy_j)
            z_numpy = np.vstack((z_numpy, z_numpy_t))
    return z_numpy


t_skeleton = np.cumsum(boomerang_kernel.dts[-num_samples:]) - boomerang_kernel.dts[-num_samples]
x_skeleton = dict_of_tensors_to_numpy(mcmc.get_samples())
v_skeleton = boomerang_kernel.v_skeleton[-num_samples:]
print(t_skeleton)


def ExtractSamples(t_skeleton, x_skeleton, v_skeleton, n_samples, x_ref, ellipticQ = True):

    T = t_skeleton[-1]
    dt = T/(n_samples-1)
    dim = x_skeleton[0].shape[0]
    t = 0.0
    x = x_skeleton[0]
    v = v_skeleton[0]
    samples = x
    skeleton_index = 2
    counter = 0
    for i in range(1, n_samples):
        t_max = (i-1) * dt
        while (counter + 1 < len(t_skeleton) and t_skeleton[counter + 1] < t_max):
            counter = counter + 1
            x = x_skeleton[counter]
            v = v_skeleton[counter]
            t = t_skeleton[counter]
        if (ellipticQ):
            (y,v) = EllipticDynamics(t_max - t, x-x_ref, v)
            x = y + x_ref
        else:
            x = x + v * (t_max - t)
        t = t_max
        samples = np.vstack((samples,x))

    return samples
def dimension_of_components(self, z):
    if type(self.key_of_z) == str:
        dimensions = [1]
    else:
        dimensions = [len(z[self.key_of_z[0]])]
        for j in range(1, len(self.key_of_z)):
            if z[self.key_of_z[j]].dim() == 0:
                dimensions.append(1)
            else:
                dimensions.append(torch.prod(
                    torch.tensor(z[self.key_of_z[j]].shape)).item())  # very rustic way to get the dimensions
    return dimensions

def shapes_of_components(self, z):
    self.shapes = {}
    for value, key in enumerate(z):
        self.shapes.update({key: z[key].shape})

def numpy_to_dict_of_tensors(self, z_numpy):
    # print("type of numpy", z_numpy.dtype)
    z_numpy = np.float32(z_numpy)
    limits = np.cumsum(self.dimensions)
    if type(self.key_of_z) == str:
        z = {self.key_of_z: torch.from_numpy(z_numpy)}
    else:
        z = {self.key_of_z[0]: torch.from_numpy(z_numpy[0:limits[0]]).reshape(self.shapes[self.key_of_z[0]])}
        for j in range(1, len(self.dimensions)):
            # convert to Numpy array
            z.update({self.key_of_z[j]: torch.from_numpy(z_numpy[limits[j - 1]: limits[j]]).reshape(
            self.shapes[self.key_of_z[j]])})
    return z


"""
samples = ExtractSamples(t_skeleton, x_skeleton, v_skeleton, 200, np.array([0,0]), ellipticQ=True)
traj = ExtractSamples(t_skeleton, x_skeleton, v_skeleton, 100000, np.array([0,0]), ellipticQ=True)
plt.scatter(skel_points[:,0], skel_points[:,1], color="red")
plt.scatter(samples[:,0], samples[:,1], color="green")
plt.plot(traj[:,0], traj[:,1], color="blue")
plt.show()
print(mcmc.get_samples()["beta"].mean(0))
print(np.mean(samples, axis=0))
print(v_skeleton)
"""