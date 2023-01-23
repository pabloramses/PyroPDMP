import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F

import pyro
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

class ZZ(MCMCKernel):

    def __init__(
            self,
            model=None,
            potential_fn=None,
            Q=None,
            transforms=None,
            dimension=None,
            excess_rate=1.0,
            ihpp_sampler=None,
            max_plate_nesting=None,
            jit_compile=False,
            jit_options=None,
            ignore_jit_warnings=False,
            init_strategy=init_to_uniform,
    ):
        if not ((model is None) ^ (potential_fn is None)):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        # NB: deprecating args - model, transforms
        self.model = model
        self.transforms = transforms
        self._max_plate_nesting = max_plate_nesting
        self._jit_compile = jit_compile
        self._jit_options = jit_options
        self._ignore_jit_warnings = ignore_jit_warnings
        self._init_strategy = init_strategy
        self.total_samp = 0.01
        self.potential_fn = potential_fn
        self.eps = np.finfo(float).eps
        self.dimensions = None
        self.key_of_z = None
        self.shapes = None
        self._no_accepted_excess_switches = 0
        self.bound = 0
        self.Q = np.abs(Q)

        self.dim = dimension
        self.excess_rate = excess_rate
        if ihpp_sampler == None:
            self.ihpp_sampler = 'Exact'
        else:
            self.ihpp_sampler = ihpp_sampler

        self._reset()
        # self._adapter = WarmupAdapter(
        #     step_size,
        #     adapt_step_size=adapt_step_size,
        #     adapt_mass_matrix=adapt_mass_matrix,
        #     target_accept_prob=target_accept_prob,
        #     dense_mass=full_mass,
        # )
        super().__init__()

    def _reset(self):  # cleans all attributes
        self._no_proposed_switches = 0
        self._no_rejected_switches = 0
        self._no_accepted_switches = 0
        self._no_refresh_switches = 0
        self._no_boundary_violated = 0
        self._t = 0
        self._accept_cnt = 0
        self._mean_accept_prob = 0.0
        self._divergences = []
        self._prototype_trace = None
        self._initial_params = None
        self._z_last = None
        self._potential_energy_last = None
        self._z_grads_last = None
        self._v_last = None
        self._warmup_steps = None

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def _initialize_model_properties(self, model_args, model_kwargs):
        init_params, potential_fn, transforms, trace = initialize_model(
            self.model,
            model_args,
            model_kwargs,
            transforms=self.transforms,
            max_plate_nesting=self._max_plate_nesting,
            jit_compile=self._jit_compile,
            jit_options=self._jit_options,
            skip_jit_warnings=self._ignore_jit_warnings,
            init_strategy=self._init_strategy,
            initial_params=self._initial_params,
        )
        self.potential_fn = potential_fn
        self.transforms = transforms
        self._initial_params = init_params
        self._prototype_trace = trace

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        if self.model is not None:
            self._initialize_model_properties(args, kwargs)
        if self.initial_params:
            z = {k: v.detach() for k, v in self.initial_params.items()}
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
        else:
            z_grads, potential_energy = {}, self.potential_fn(self.initial_params)
        # Initiate a velocity
        initial_v = np.random.normal(0, 1, self.dim)
        self._cache(self.initial_params, initial_v, potential_energy, z_grads)

    def cleanup(self):
        self._reset()

    def _cache(self, z, v, potential_energy, z_grads=None):
        self._z_last = z
        self._v_last = v
        self._potential_energy_last = potential_energy
        self._z_grads_last = z_grads

    def clear_cache(self):
        self._z_last = None
        self._v_last = None
        self._potential_energy_last = None
        self._z_grads_last = None

    def _fetch_from_cache(self):
        return self._z_last, self._v_last, self._potential_energy_last, self._z_grads_last

    def logging(self):
        # return None
        return OrderedDict(
            [
                ("prop. viol",
                 "{:.3f}".format(self._no_boundary_violated / self._no_proposed_switches)),
                (
                "acc prop", "{:.3f}".format(self._no_accepted_switches / self._no_proposed_switches)),
                ("acc step", "{:.3f}".format(self._no_accepted_switches / self.total_samp)),
                ("Switch time proposed", self.dt_switch_proposed),
                ("Bound", self.bound),
            ]
        )

    def diagnostics(self):
        # return {}
        return {
            # "divergences": self._divergences,
            "no of boundary violations": self._no_boundary_violated,
            "prop. of accepted switches": self._no_accepted_switches / self._no_proposed_switches,
            "prop. of accepted switches due to excess": self._no_refresh_switches / self._no_accepted_switches,
        }

    def sample(self, params):
        z, v, potential_energy, z_grads = self._fetch_from_cache()
        # recompute PE when cache is cleared
        if z is None:
            z = params
            v = np.random.choice([-1, 1], size=self.dim)
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
            self._cache(z, v, potential_energy, z_grads)
        # Extract key of z
        if self.key_of_z == None:
            self.extract_keys_of_z(z)  # extracts the keys of z just the first time it samples
        if self.dimensions == None:  # only first time
            self.dimensions = self.dimension_of_components(
                z)  # collects the dimension of each component of the model (dimension of value of each key)
        if self.shapes == None:
            self.shapes_of_components(z)

        # convert to Numpy array
        z_numpy = self.dict_of_tensors_to_numpy(z)
        z_grads_numpy = self.dict_of_tensors_to_numpy(z_grads)

        # denote time as zero
        t = 0.0
        updateSkeleton = False
        finished = False

        if (self.excess_rate == 0.0):
            dt_excess = np.Inf
        else:
            dt_excess = -np.log(np.random.rand()) / (self.dim * self.excess_rate)
        if self.ihpp_sampler == 'Exact':
            # Compute switching time
            #b = self.hessian_bound * np.ones(self.dim)
            #b = np.dot(self.Q, np.ones(self.dim))
            #matricina = np.matmul(self.Q, np.eye(self.dim))
            b = np.array([np.linalg.norm(self.Q[:,i]) for i in range(self.Q.shape[1])])*self.dim
            #b = np.dot(self.Q, np.ones(self.dim)) + 1000000
            a = np.multiply(z_grads_numpy, v)
            dt_proposed_switches = [self.switchingtime(a[i], b[i]) for i in range(len(a))]
            while not finished:
                i = np.argmin(dt_proposed_switches)  # O(d)
                dt_switch_proposed = dt_proposed_switches[i]
                self.dt_switch_proposed = dt_switch_proposed
                dt = min(dt_switch_proposed, dt_excess)
                self._no_proposed_switches = self._no_proposed_switches + 1
                # T = 10
                # if t + Δt > T:
                #     Δt = T - t
                #     finished = True
                #     updateSkeleton = True

                z_numpy = z_numpy + v * dt  # O(d)
                # Convert to tensor to save and to compute gradient
                z = self.numpy_to_dict_of_tensors(z_numpy)
                z_grads_new, potential_energy_new = potential_grad(self.potential_fn, z)
                # convert z_grads_new to numpy
                z_grads_new_numpy = self.dict_of_tensors_to_numpy(z_grads_new)
                t = t + dt
                a = a + b * dt  # O(d)
                self.bound = a[i]

                if not finished and dt_switch_proposed < dt_excess:
                    switch_rate = v[i] * z_grads_new_numpy[i]
                    proposedSwitchIntensity = a[i]
                    if proposedSwitchIntensity < switch_rate:
                        self._no_boundary_violated = self._no_boundary_violated + 1


                    if np.random.random() * proposedSwitchIntensity <= switch_rate:
                        self._no_accepted_switches = self._no_accepted_switches + 1
                        # switch i-th component
                        v[i] = -v[i]
                        a[i] = -switch_rate
                        updateSkeleton = True
                        finished = True
                    else:
                        a[i] = switch_rate
                        updateSkeleton = False
                        self._no_rejected_switches = self._no_rejected_switches + 1

                    # update refreshment time and switching time bound
                    dt_excess = dt_excess - dt_switch_proposed
                    dt_proposed_switches = dt_proposed_switches - dt_switch_proposed
                    dt_proposed_switches[i] = self.switchingtime(a[i], b[i])
                elif not finished:
                    # so we switch due to excess switching rate
                    updateSkeleton = True
                    finished = True
                    i = np.random.choice(range(0, self.dim), size=1)
                    v[i] = -v[i]
                    a[i] = v[i] * z_grads_new_numpy[i]
                    self._no_accepted_excess_switches = self._no_accepted_excess_switches + 1

                    # update upcoming event times
                    dt_proposed_switches = dt_proposed_switches - dt_excess
                    dt_excess = -np.log(np.random.rand()) / (self.dim * self.excess_rate)

                if updateSkeleton:
                    self.total_samp = self.total_samp + 1
                    updateSkeleton = False
                    self._cache(z, v, potential_energy_new, z_grads_new)

        elif self.ihpp_sampler == 'Corbella':
            rebound = True
            while not finished:
                if rebound:
                    arg, a = self.corbella(z_numpy, v, dt_excess)
                    self.bound = a
                    rebound = False
                if a == 0:
                    dt_switch_proposed = 1e16
                else:
                    dt_switch_proposed = self.switchingtime(a, 0)
                self.dt_switch_proposed = dt_switch_proposed
                dt = np.minimum(dt_switch_proposed, dt_excess)
                self._no_proposed_switches = self._no_proposed_switches + 1
                # Update z and v
                (y, v) = self.Dynamics(dt, z_numpy, v)
                z_numpy = y

                # Convert to tensor to save and to compute gradient
                z = self.numpy_to_dict_of_tensors(z_numpy)
                z_grads_new, potential_energy_new = potential_grad(self.potential_fn, z)
                # grads_new to numpy
                z_grads_new_numpy = self.dict_of_tensors_to_numpy(z_grads_new)
                gradU = z_grads_new_numpy
                t = t + dt
                if not finished and dt_switch_proposed < dt_excess:
                    switch_rate = np.dot(v, gradU)  # no need to take positive part
                    simulated_rate = a
                    if simulated_rate < switch_rate:
                        self._no_boundary_violated = self._no_boundary_violated + 1
                        # print("simulated rate: ", simulated_rate)
                        # print("actual switching rate: ", switch_rate)
                        # print("switching rate exceeds bound.")
                        # Should not we raise value error?
                        # raise ValueError("Switching rate exceeds bound.")

                    if np.random.rand() * simulated_rate <= switch_rate:
                        # sample the component that will be switched
                        un_probs = [np.maximum(0, v[i] * gradU[i]) for i in range(self.dim)]
                        probs = un_probs / np.sum(un_probs)
                        i = np.random.choice(np.arange(0, self.dim), p=probs)
                        v[i] = -v[i]
                        updateSkeleton = True
                        finished = True
                        rebound = True
                        self._no_accepted_switches = self._no_accepted_switches + 1
                    else:
                        updateSkeleton = False
                        self._no_rejected_switches = self._no_rejected_switches + 1

                    # update refreshment time and switching time bound
                    dt_excess = dt_excess - dt_switch_proposed


                elif not finished:
                    # so we refresh
                    self._no_refresh_switches = self._no_refresh_switches + 1
                    updateSkeleton = True
                    finished = True
                    rebound = True
                    i = np.random.choice(np.arange(0, self.dim - 1))
                    v[i] = -v[i]

                    # compute new refreshment time
                    dt_excess = -np.log(np.random.rand()) / self.excess_rate
                else:
                    pass

                if updateSkeleton:
                    self.total_samp = self.total_samp + 1
                    # self._no_accepted_switches = self._no_accepted_switches + 1
                    updateSkeleton = False
                    self._cache(z, v, potential_energy_new, z_grads_new)

        return z.copy()

    def extract_keys_of_z(self, z):
        if len(list(z.keys())) == 1:
            self.key_of_z = list(z.keys())[0]
        else:
            self.key_of_z = list(z.keys())

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

    def dict_of_tensors_to_numpy(self, z):
        # print("type of initial dicti",z[self.key_of_z[0]].dtype)
        if len(list(z.keys())) == 1:
            self.key_of_z = list(z.keys())[0]
            z_numpy = z[self.key_of_z].numpy()
        else:
            self.key_of_z = list(z.keys())
            z_numpy = z[self.key_of_z[0]].numpy()
            for j in range(1, len(self.key_of_z)):
                # convert to Numpy array
                z_numpy_j = z[self.key_of_z[j]].numpy()
                z_numpy = np.append(z_numpy, z_numpy_j)
        # print("type of numpy", z_numpy.dtype)
        return z_numpy

    def numpy_to_dict_of_tensors(self, z_numpy):
        # print("type of numpy", z_numpy.dtype)
        z_numpy = np.float32(z_numpy)
        limits = np.cumsum(self.dimensions)
        if type(self.key_of_z) == str:
            z = {self.key_of_z: torch.from_numpy(z_numpy)}
            z_grads_new, potential_energy_new = potential_grad(self.potential_fn, z)
        else:
            z = {self.key_of_z[0]: torch.from_numpy(z_numpy[0:limits[0]]).reshape(self.shapes[self.key_of_z[0]])}
            for j in range(1, len(self.dimensions)):
                # convert to Numpy array
                z.update({self.key_of_z[j]: torch.from_numpy(z_numpy[limits[j - 1]: limits[j]]).reshape(
                    self.shapes[self.key_of_z[j]])})
        # print("type of posterior dict", z[self.key_of_z[0]].dtype)
        return z

    def rate_of_t(self, z, v, t):
        zt_numpy, vt = self.Dynamics(t, z, v)
        zt = self.numpy_to_dict_of_tensors(zt_numpy)
        z_grads_new, potential_energy_new = potential_grad(self.potential_fn, zt)
        z_grads_new_numpy = self.dict_of_tensors_to_numpy(z_grads_new)
        gradU = z_grads_new_numpy
        ft = np.dot(vt, gradU)
        return np.maximum(0, ft)

    def corbella(self, z, v, tmax, check=True):
        def minus_rate_of_t(t):
            zt_numpy, vt = self.Dynamics(t, z, v)
            zt = self.numpy_to_dict_of_tensors(zt_numpy)
            z_grads_new, potential_energy_new = potential_grad(self.potential_fn, zt)
            z_grads_new_numpy = self.dict_of_tensors_to_numpy(z_grads_new)
            gradU = z_grads_new_numpy
            ft = -np.dot(vt, gradU)
            return np.minimum(0, ft)

        if check == False:
            argmin = optimize.fminbound(minus_rate_of_t, 0, tmax, xtol=1.48e-08, full_output=0, maxfun=100)
        else:
            candidate, fval, ierr, numit = optimize.fminbound(minus_rate_of_t, 0, tmax, xtol=1.48e-08, full_output=1,
                                                              maxfun=1, disp=0)
            if ierr == 0:
                argmin = candidate
            elif (minus_rate_of_t(0) >= fval >= minus_rate_of_t(tmax) and minus_rate_of_t(0) >= minus_rate_of_t(
                    self.eps)):  # ->the rate might be monotonically non-decreasing
                argmin = tmax
            elif (minus_rate_of_t(tmax) >= fval >= minus_rate_of_t(0) and minus_rate_of_t(tmax) <= minus_rate_of_t(
                    tmax - self.eps)):  # ->the rate might be monotonically non-increasing
                argmin = 0
            else:
                argmin = optimize.fminbound(minus_rate_of_t, 0, tmax, xtol=1.48e-08, full_output=0, maxfun=100)
        return argmin, -minus_rate_of_t(argmin)

    def switchingtime(self, a, b, u=None):
        # generate switching time for rate of the form max(0, a + b s) + c
        # under the assumptions that b > 0, c > 0
        # u is the random number
        if u:
            pass
        else:
            u = np.random.rand()
        if (b > 0):
            if (a < 0):
                return -a / b + self.switchingtime(0.0, b, u)
            else:  # a >= 0
                return -a / b + np.sqrt(np.power(a, 2) / np.power(b, 2) - 2 * np.log(u) / b)
        elif (b == 0):  # degenerate case
            if (a < 0):
                return np.Inf
            else:  # a >= 0
                return -np.log(u) / a
        else:  # b <= 0
            print('b neg')
            if (a <= 0):
                return np.Inf
            else:  # a > 0
                y = -np.log(u)
                t1 = -a / b
                if (y >= a * t1 + b * np.power(t1, 2) / 2):
                    return np.Inf
                else:
                    return -a / b - np.sqrt(np.power(a, 2) / np.power(b, 2) + 2 * y / b)

    def Dynamics(self, t, y0, w0):
        # simulate dy/dt = w, dw/dt = -y
        y_new = y0 + w0 * t
        w_new = w0
        return (y_new, w_new)