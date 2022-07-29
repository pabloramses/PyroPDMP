# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
from collections import OrderedDict

import torch

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


class ZZ(MCMCKernel):
    r"""
    Simple Hamiltonian Monte Carlo kernel, where ``step_size`` and ``num_steps``
    need to be explicitly specified by the user.
    **References**
    [1] `MCMC Using Hamiltonian Dynamics`,
    Radford M. Neal
    :param model: Python callable containing Pyro primitives.
    :param potential_fn: Python callable calculating potential energy with input
        is a dict of real support parameters.
    :param nd-array Q: is a symmetric matrix with nonnegative entries such
        that |(∇^2 E(x))_{ij}| <= Q_{ij} for all x, i, j
    :param float excess_rate: excess_rate
    :param dict transforms: Optional dictionary that specifies a transform
        for a sample site with constrained support to unconstrained space. The
        transform should be invertible, and implement `log_abs_det_jacobian`.
        If not specified and the model has sites with constrained support,
        automatic transformations will be applied, as specified in
        :mod:`torch.distributions.constraint_registry`.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. This is required if model contains
        discrete sample sites that can be enumerated over in parallel.
    :param bool jit_compile: Optional parameter denoting whether to use
        the PyTorch JIT to trace the log density computation, and use this
        optimized executable trace in the integrator.
    :param dict jit_options: A dictionary contains optional arguments for
        :func:`torch.jit.trace` function.
    :param bool ignore_jit_warnings: Flag to ignore warnings from the JIT
        tracer when ``jit_compile=True``. Default is False.
    :param float target_accept_prob: Increasing this value will lead to a smaller
        step size, hence the sampling will be slower and more robust. Default to 0.8.
    :param callable init_strategy: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    .. note:: Internally, the mass matrix will be ordered according to the order
        of the names of latent variables, not the order of their appearance in
        the model.
    Example:
        # >>> true_coefs = torch.tensor([1., 2., 3.])
        # >>> data = torch.randn(2000, 3)
        # >>> dim = 3
        # >>> labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()
        # >>>
        # >>> def model(data):
        # ...     coefs_mean = torch.zeros(dim)
        # ...     coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))
        # ...     y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        # ...     return y
        # >>>
        # >>> hmc_kernel = HMC(model, step_size=0.0855, num_steps=4)
        # >>> mcmc = MCMC(hmc_kernel, num_samples=500, warmup_steps=100)
        # >>> mcmc.run(data)
        # >>> mcmc.get_samples()['beta'].mean(0)  # doctest: +SKIP
        # tensor([ 0.9819,  1.9258,  2.9737])
    """

    def __init__(
        self,
        model=None,
        potential_fn=None,
        Q = None,
        excess_rate = 0.0,
        transforms=None,
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

        self.potential_fn = potential_fn

        # Some inputs specific for ZigZag
        self.Q = Q #np.array([[2000,2000],[2000,2000]])
        self.dim = self.Q.shape[0]
        self.excess_rate = excess_rate

        self._reset()
        # self._adapter = WarmupAdapter(
        #     step_size,
        #     adapt_step_size=adapt_step_size,
        #     adapt_mass_matrix=adapt_mass_matrix,
        #     target_accept_prob=target_accept_prob,
        #     dense_mass=full_mass,
        # )
        super().__init__()

    def _reset(self):
        self._no_proposed_switches = 0
        self._no_rejected_switches = 0
        self._no_accepted_switches = 0
        self._no_accepted_excess_switches = 0
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
        initial_v = np.random.choice([-1, 1], size=self.dim)
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
        #return None
        return OrderedDict(
            [
                ("prop. of boundary violation", "{:.3f}".format(self._no_boundary_violated / self._no_proposed_switches)),
                ("prop. of accepted switches", "{:.3f}".format(self._no_accepted_switches / self._no_proposed_switches)),
            ]
        )

    def diagnostics(self):
        #return {}
        return {
            #"divergences": self._divergences,
            "no of boundary violations": self._no_boundary_violated,
            "prop. of accepted switches": self._no_accepted_switches / self._no_proposed_switches,
            "prop. of accepted switches due to excess": self._no_accepted_excess_switches / self._no_accepted_switches,
        }

    def switchingtime(self, a, b, u=np.random.random()):
        # generate switching time for rate of the form max(0, a + b s) + c
        # under the assumptions that b > 0, c > 0
        # u is the random number
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

    def sample(self, params):
        z, v, potential_energy, z_grads = self._fetch_from_cache()
        # recompute PE when cache is cleared
        if z is None:
            z = params
            v = np.random.choice([-1, 1], size=self.dim)
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
            self._cache(z, v, potential_energy, z_grads)
        # Extract key of z
        key_of_z = list(z.keys())[0]

        # convert to Numpy array
        z_numpy, z_grads_numpy = z[key_of_z].numpy(), z_grads[key_of_z].numpy()

        # denote time as zero
        t = 0
        updateSkeleton = False
        finished = False

        # Compute switching time
        b = np.dot(self.Q, np.ones(shape=(self.dim,)))
        a = np.multiply(z_grads_numpy, v)
        Δt_proposed_switches = [self.switchingtime(a[i], b[i]) for i in range(len(a))]
        if (self.excess_rate == 0.0):
            Δt_excess = np.Inf
        else:
            Δt_excess = -np.log(np.random())/(self.dim*self.excess_rate)

        while not finished:
            i = np.argmin(Δt_proposed_switches)  # O(d)
            Δt_switch_proposed = Δt_proposed_switches[i]
            Δt = min(Δt_switch_proposed, Δt_excess)
            self._no_proposed_switches = self._no_proposed_switches + 1
            # T = 10
            # if t + Δt > T:
            #     Δt = T - t
            #     finished = True
            #     updateSkeleton = True

            z_numpy = z_numpy + v * Δt  # O(d)
            # Convert to tensor to save and to compute gradient
            z = {key_of_z:torch.from_numpy(z_numpy)}
            z_grads_new, potential_energy_new = potential_grad(self.potential_fn, z)
            # convert z_grads_new to numpy
            z_grads_new_numpy = z_grads_new[key_of_z].numpy()
            t = t + Δt
            a = a + b * Δt  # O(d)

            if not finished and Δt_switch_proposed < Δt_excess:
                switch_rate = v[i] * z_grads_new_numpy[i]
                proposedSwitchIntensity = a[i]
                if proposedSwitchIntensity < switch_rate:
                    self._no_boundary_violated = self._no_boundary_violated + 1
                    print("ERROR: Switching rate exceeds bound.")
                    print(" simulated rate: ", proposedSwitchIntensity)
                    print(" actual switching rate: ", switch_rate)
                    #raise ValueError("Switching rate exceeds bound.")
                    # error()

                if np.random.random() * proposedSwitchIntensity <= switch_rate:
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
                Δt_excess = Δt_excess - Δt_switch_proposed
                Δt_proposed_switches = Δt_proposed_switches - Δt_switch_proposed
                Δt_proposed_switches[i] = self.switchingtime(a[i], b[i])
            elif not finished:
                # so we switch due to excess switching rate
                updateSkeleton = True
                finished = True
                i = np.random.choice(range(1, self.dim + 1), size=1)
                v[i] = -v[i]
                a[i] = v[i] * z_grads_new[i]
                self._no_accepted_excess_switches = self._no_accepted_excess_switches + 1

                # update upcoming event times
                Δt_proposed_switches = Δt_proposed_switches - Δt_excess
                Δt_excess = -np.log(np.random()) / (self.dim * self.excess_rate)

            if updateSkeleton:
                self._no_accepted_switches = self._no_accepted_switches + 1
                updateSkeleton = False
                self._cache(z, v, potential_energy_new, z_grads_new)

        return z.copy()


true_coefs = torch.tensor([1., 2.])
data = torch.randn(2000, 2)
dim = 2
labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()
def model(data):
     coefs_mean = torch.zeros(dim)
     coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(2)))
     y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
     return y
zz_kernel = ZZ(model, Q=np.array([[2000,2000],[2000,2000]]), excess_rate=0.0)
from pyro.infer import MCMC
mcmc = MCMC(zz_kernel, num_samples=5000)
mcmc.run(data)
print(mcmc.get_samples()['beta'].mean(0))