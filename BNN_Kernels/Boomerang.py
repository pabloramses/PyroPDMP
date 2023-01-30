import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.distributions.testing.fakes import NonreparameterizedNormal
from pyro.distributions.util import scalar_like
from pyro.infer.autoguide import init_to_uniform, init_to_sample, init_to_value
from pyro.infer.mcmc.adaptation import WarmupAdapter
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from pyro.ops.integrator import potential_grad, velocity_verlet
from pyro.util import optional, torch_isnan
from scipy import optimize
from scipy import integrate

"STANDARD BOOMERANG"
class Boomerang(MCMCKernel):

    def __init__(
            self,
            model=None,
            potential_fn=None,
            hessian_bound=None,
            transforms=None,
            Sigma=None,
            refresh_rate=1.0,
            batch_size=None,
            shuffle=True,
            max_plate_nesting=None,
            jit_compile=False,
            jit_options=None,
            ignore_jit_warnings=False,
            init_strategy=init_to_sample,
            parameter_list = None,
            hyperparameter_list = None,
            gibbs_rate = 1,
            RW_scale = None,
            initial_parameters=None,
            list_of_layers=None,
            list_of_types_of_parametres=None,
            t_max = None
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
        self.t = 0.0
        self.dts = [0.0]
        self.move = False
        if batch_size is None:
            self.subsampling = False
        else:
            self.subsampling = True
            self.batch_size = batch_size
            self.shuffle = shuffle


        # Some inputs specific for Boomerang
        self.Sigma = Sigma  # np.array([[3,0.5],[0.5,3]])
        self.dim = self.Sigma.shape[0]
        self.z_ref = np.zeros(self.dim)  # mean of reference measure
        if hessian_bound == None:
            self.Q = np.linalg.norm(
                np.linalg.inv(self.Sigma))  # currently -> bound on the hessian of the energy for gaussian
        else:
            self.Q = hessian_bound
        # self.Q = np.array([[2000,2000],[2000,2000]])
        self.refresh_rate = refresh_rate

        #gibbs setting
        if parameter_list is not None:
            self.parameter_list = parameter_list
            self.hyperparameter_list = hyperparameter_list
            self.gibbs_rate = gibbs_rate
            self.gibbs = True
        else:
            self.gibbs = False
        #random walk
        if self.gibbs:
            self.RW = False

            if RW_scale is not None:
                self.RW = True
                self.scale = RW_scale
            if list_of_layers is not None:
                self.layers = list_of_layers
            if list_of_types_of_parametres is not None:
                self.types_of_parameters = list_of_types_of_parametres

        self.parametros_iniciales = initial_parameters
        self.t_max = t_max
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
        self._no_accepted_switches = 0.1
        self._no_accepted_gibbs = 0
        self._no_proposed_gibbs = 0.01
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
        self._initial_params = self.parametros_iniciales
        self._prototype_trace = trace

    def setup(self, warmup_steps, *args, **kwargs):
        self._args = args
        self._warmup_steps = warmup_steps
        if self.model is not None:
            self._initialize_model_properties(args, kwargs)
        if self.initial_params:
            z = {k: v.detach() for k, v in self.initial_params.items()}
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
        else:
            z_grads, potential_energy = {}, self.potential_fn(self.initial_params)
        # Initiate a velocity
        #print(z)
        initial_v = np.dot(np.linalg.cholesky(self.Sigma), np.random.normal(0, 1, self.dim))
        self.v_skeleton = initial_v
        self._cache(self.initial_params, initial_v, potential_energy, z_grads)

    def recalculate_potential(self, sample, parameters):
        params, potential_fn, transforms, trace = initialize_model(
            self.model,
            sample,
            model_kwargs={},
            transforms=self.transforms,
            max_plate_nesting=self._max_plate_nesting,
            jit_compile=self._jit_compile,
            jit_options=self._jit_options,
            skip_jit_warnings=self._ignore_jit_warnings,
            init_strategy=init_to_value(values=parameters),
            initial_params=parameters,
        )
        self.potential_fn = potential_fn
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
                ("pv","{:.3f}".format(self._no_boundary_violated / self._no_proposed_switches)),
                ("ihpp_ar", "{:.3f}".format(self._no_accepted_switches / self._no_proposed_switches)),
                ("mh_ar", "{:.3f}".format(self._no_accepted_gibbs / self._no_proposed_gibbs)),
                ("gib step", "{:.3f}".format(self._no_proposed_gibbs / self.total_samp)),
                ("acc step", "{:.3f}".format(self._no_accepted_switches / self.total_samp)),
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
        if self.gibbs:
            z, v, potential_energy, z_grads = self._fetch_from_cache()

            if self.subsampling:
                x, y = self.subsample()
                self.recalculate_potential((x,y), z)
            # recompute PE when cache is cleared
            if z is None:
                z = params
                v = np.dot(np.linalg.cholesky(self.Sigma), np.random.normal(0, 1, self.dim))
                z_grads, potential_energy = potential_grad(self.potential_fn, z)
                self._cache(z, v, potential_energy, z_grads)
            # Extract key of z
            z_parameters, z_hyperparameters = self.split_param_hyper(z)
            z_grads_parameters, z_grads_hyperparameters = self.split_param_hyper(z_grads)
            if self.key_of_z == None:
                self.extract_keys_of_z(z)  # extracts the keys of z just the first time it samples
            if self.dimensions == None:  # only first time
                self.dimensions = self.dimension_of_components(
                    z_parameters, self.parameter_list)  # collects the dimension of each component of the model (dimension of value of each key)
            if self.shapes == None:
                self.shapes = self.shapes_of_components(z_parameters)
                self.shapes_hyperparametres = self.shapes_of_components(z_hyperparameters)




            z_parameters_numpy = self.dict_of_tensors_to_numpy(z_parameters)
            z_grads_parameters_numpy = self.dict_of_tensors_to_numpy(z_grads_parameters)
            z_hyperparameters_numpy = self.dict_of_tensors_to_numpy(z_hyperparameters)
            z_grads_hyperparameters_numpy = self.dict_of_tensors_to_numpy(z_grads_hyperparameters)
            t = 0.0
            # Modified gradU
            updateSkeleton = False
            finished = False

            dt_refresh = -np.log(np.random.rand()) / self.refresh_rate
            dt_gibbs = -np.log(np.random.rand()) / self.gibbs_rate
            dt_limit = np.minimum(dt_gibbs, dt_gibbs)
            subintervals = 2
            t_max = dt_limit/subintervals

            rebound = True
            while not finished:
                if rebound:
                    arg, a = self.corbella(z_parameters_numpy, v, t_max, z_hyperparameters=z_hyperparameters)
                    self.bound = a
                    rebound = False
                if a == 0:
                    dt_switch_proposed = 1e16
                else:

                    dt_switch_proposed = self.switchingtime(a, 0)

                dt = np.min(np.array([t_max, dt_switch_proposed, dt_refresh, dt_gibbs]))
                self._no_proposed_switches = self._no_proposed_switches + 1
                # Update z and v
                (y, v) = self.EllipticDynamics(dt, z_parameters_numpy - self.z_ref, v)
                z_parameters_numpy = y + self.z_ref

                # Convert to tensor to save and to compute gradient
                z_parameters = self.numpy_to_dict_of_tensors(z_parameters_numpy, self.parameter_list)

                z_rightlim = self.merge_param_hyper(z_parameters, z_hyperparameters)
                z_grads_new, potential_energy_new = potential_grad(self.potential_fn, z_rightlim)
                z_parameters_grads_new, z_hyperparameters_grads_new = self.split_param_hyper(z_grads_new)
                # grads_new to numpy
                z_parameters_grads_new_numpy = self.dict_of_tensors_to_numpy(z_parameters_grads_new)
                gradU = z_parameters_grads_new_numpy - np.dot(np.linalg.inv(self.Sigma), (z_parameters_numpy - self.z_ref))
                t = t + dt
                if not finished and dt_switch_proposed < dt_refresh and dt_switch_proposed < dt_gibbs and dt_switch_proposed < t_max:
                    switch_rate = np.dot(v, gradU)  # no need to take positive part
                    simulated_rate = a
                    if simulated_rate < switch_rate:
                        self._no_boundary_violated = self._no_boundary_violated + 1


                    if np.random.rand() * simulated_rate <= switch_rate:
                        # obtain new velocity by reflection
                        skewed_grad = np.dot(np.transpose(np.linalg.cholesky(self.Sigma)), gradU)
                        v = v - 2 * (switch_rate / np.dot(skewed_grad, skewed_grad)) * np.dot(
                            np.linalg.cholesky(self.Sigma), skewed_grad)
                        updateSkeleton = True
                        self.move = True
                        finished = True
                        rebound = True
                        self._no_accepted_switches = self._no_accepted_switches + 1
                    else:
                        updateSkeleton = False
                        self.move = False
                        self._no_rejected_switches = self._no_rejected_switches + 1
                    # update refreshment time and switching time bound
                    dt_refresh = dt_refresh - dt_switch_proposed
                    dt_gibbs = dt_gibbs - dt_switch_proposed

                elif not finished and t_max <=dt_refresh and t_max <=dt_gibbs and t_max<dt_switch_proposed:
                    updateSkeleton = False
                    self.move = False
                    rebound = True
                    # update refreshment time and switching time bound
                    dt_refresh = dt_refresh - t_max
                    dt_gibbs = dt_gibbs - t_max

                elif not finished and dt_refresh < dt_switch_proposed and dt_refresh < dt_gibbs and dt_refresh < t_max:
                    # so we refresh
                    self._no_refresh_switches = self._no_refresh_switches + 1
                    updateSkeleton = True
                    self.move = True
                    finished = True
                    rebound = True
                    v = np.dot(np.linalg.cholesky(self.Sigma), np.random.normal(0, 1, self.dim))

                    # compute new refreshment time
                    dt_refresh = -np.log(np.random.rand()) / self.refresh_rate
                elif not finished and dt_gibbs < dt_switch_proposed and dt_gibbs < dt_refresh and dt_gibbs < t_max:
                    #update the hyperparameters
                    rebound = True
                    finished = True
                    updateSkeleton = True
                    self._no_proposed_gibbs = self._no_proposed_gibbs + 1
                    z_hyperparameters = self.rwmh_HP(z_parameters, z_hyperparameters, scale=self.scale)
                    dt_gibbs = -np.log(np.random.rand()) / self.gibbs_rate
                if updateSkeleton:
                    z = self.merge_param_hyper(z_parameters, z_hyperparameters)
                    self.total_samp = self.total_samp + 1
                    # self._no_accepted_switches = self._no_accepted_switches + 1
                    updateSkeleton = False
                    self.v_skeleton = np.vstack((self.v_skeleton, np.transpose(v)))
                    self._cache(z, v, potential_energy_new, z_grads_new)
                    self.t = self.t + dt
                    self.dts.append(dt)

        else:
            z, v, potential_energy, z_grads = self._fetch_from_cache()

            if self.subsampling:
                x, y = self.subsample()
                self.recalculate_potential((x, y), z)
            # recompute PE when cache is cleared
            if z is None:
                z = params
                v = np.dot(np.linalg.cholesky(self.Sigma), np.random.normal(0, 1, self.dim))
                z_grads, potential_energy = potential_grad(self.potential_fn, z)
                self._cache(z, v, potential_energy, z_grads)
            # Extract key of z
            #z_parameters, z_hyperparameters = self.split_param_hyper(z)
            #z_grads_parameters, z_grads_hyperparameters = self.split_param_hyper(z_grads)
            if self.key_of_z == None:
                self.extract_keys_of_z(z)  # extracts the keys of z just the first time it samples
            if self.dimensions == None:  # only first time
                self.dimensions = self.dimension_of_components(
                    z,
                    self.parameter_list)  # collects the dimension of each component of the model (dimension of value of each key)
            if self.shapes == None:
                self.shapes = self.shapes_of_components(z)
                #self.shapes_hyperparametres = self.shapes_of_components(z_hyperparameters)

            z_numpy = self.dict_of_tensors_to_numpy(z)
            z_grads_numpy = self.dict_of_tensors_to_numpy(z_grads)
            t = 0.0
            # Modified gradU
            updateSkeleton = False
            finished = False

            dt_refresh = -np.log(np.random.rand()) / self.refresh_rate
            dt_gibbs = -np.log(np.random.rand()) / self.gibbs_rate

            rebound = True
            while not finished:
                if rebound:
                    arg, a = self.corbella(z_numpy, v, dt_refresh)
                    self.bound = a
                    rebound = False
                if a == 0:
                    dt_switch_proposed = 1e16
                else:
                    dt_switch_proposed = self.switchingtime(a, 0)
                self.dt_switch_proposed = dt_switch_proposed
                dt = np.min(np.array([dt_switch_proposed, dt_refresh, dt_gibbs]))
                self._no_proposed_switches = self._no_proposed_switches + 1
                # Update z and v
                (y, v) = self.EllipticDynamics(dt, z_numpy - self.z_ref, v)
                z_numpy = y + self.z_ref

                # Convert to tensor to save and to compute gradient
                z = self.numpy_to_dict_of_tensors(z_numpy, self.key_of_z)

                #z_rightlim = self.merge_param_hyper(z_parameters, z_hyperparameters)
                z_grads_new, potential_energy_new = potential_grad(self.potential_fn, z)
                #z_parameters_grads_new, z_hyperparameters_grads_new = self.split_param_hyper(z_grads_new)
                # grads_new to numpy
                z_grads_new_numpy = self.dict_of_tensors_to_numpy(z_grads_new)
                gradU = z_grads_new_numpy - np.dot(np.linalg.inv(self.Sigma),
                                                              (z_numpy - self.z_ref))
                t = t + dt
                if not finished and dt_switch_proposed < dt_refresh:
                    switch_rate = np.dot(v, gradU)  # no need to take positive part
                    simulated_rate = a
                    if simulated_rate < switch_rate:
                        self._no_boundary_violated = self._no_boundary_violated + 1

                    if np.random.rand() * simulated_rate <= switch_rate:
                        # obtain new velocity by reflection
                        skewed_grad = np.dot(np.transpose(np.linalg.cholesky(self.Sigma)), gradU)
                        v = v - 2 * (switch_rate / np.dot(skewed_grad, skewed_grad)) * np.dot(
                            np.linalg.cholesky(self.Sigma), skewed_grad)
                        updateSkeleton = True
                        self.move = True
                        finished = True
                        rebound = True
                        self._no_accepted_switches = self._no_accepted_switches + 1
                    else:
                        updateSkeleton = False
                        self.move = False
                        self._no_rejected_switches = self._no_rejected_switches + 1

                    # update refreshment time and switching time bound
                    dt_refresh = dt_refresh - dt_switch_proposed


                elif not finished and dt_refresh < dt_switch_proposed:
                    # so we refresh
                    self._no_refresh_switches = self._no_refresh_switches + 1
                    updateSkeleton = True
                    self.move = True
                    finished = True
                    rebound = True
                    v = np.dot(np.linalg.cholesky(self.Sigma), np.random.normal(0, 1, self.dim))

                    # compute new refreshment time
                    dt_refresh = -np.log(np.random.rand()) / self.refresh_rate

                if updateSkeleton:
                    self.total_samp = self.total_samp + 1
                    # self._no_accepted_switches = self._no_accepted_switches + 1
                    updateSkeleton = False
                    self.v_skeleton = np.vstack((self.v_skeleton, np.transpose(v)))
                    self._cache(z, v, potential_energy_new, z_grads_new)
                    self.t = self.t + dt
                    self.dts.append(dt)


        return z.copy()

    def extract_keys_of_z(self, z):
        if len(list(z.keys())) == 1:
            self.key_of_z = list(z.keys())[0]
        else:
            self.key_of_z = list(z.keys())

    def dimension_of_components(self, z, keys):
        if type(keys) == str:
            dimensions = [1]
        else:
            dimensions = [len(z[keys[0]])]
            for j in range(1, len(keys)):
                if z[keys[j]].dim() == 0:
                    dimensions.append(1)
                else:
                    dimensions.append(torch.prod(
                        torch.tensor(z[keys[j]].shape)).item())  # very rustic way to get the dimensions
        return dimensions

    def shapes_of_components(self, z):
        shapes = {}
        for value, key in enumerate(z):
            shapes.update({key: z[key].shape})
        return shapes

    def dict_of_tensors_to_numpy(self, z):
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
        return z_numpy

    def numpy_to_dict_of_tensors(self, z_numpy, keys):
        z_numpy = np.float32(z_numpy)
        limits = np.cumsum(self.dimensions)
        if type(keys) == str:
            z = {keys: torch.from_numpy(z_numpy)}
            z_grads_new, potential_energy_new = potential_grad(self.potential_fn, z)
        else:
            z = {keys[0]: torch.from_numpy(z_numpy[0:limits[0]]).reshape(self.shapes[keys[0]])}
            for j in range(1, len(self.dimensions)):
                # convert to Numpy array
                z.update({keys[j]: torch.from_numpy(z_numpy[limits[j - 1]: limits[j]]).reshape(
                    self.shapes[keys[j]])})
        return z

    def rate_of_t(self, z, v, t):
        zt_numpy, vt = self.EllipticDynamics(t, z, v)
        zt = self.numpy_to_dict_of_tensors(zt_numpy, self.key_of_z)
        z_grads_new, potential_energy_new = potential_grad(self.potential_fn, zt)
        z_grads_new_numpy = self.dict_of_tensors_to_numpy(z_grads_new)
        gradU = z_grads_new_numpy - np.dot(np.linalg.inv(self.Sigma), (zt_numpy - self.z_ref))
        ft = np.dot(vt, gradU)
        return np.maximum(0, ft)

    def corbella(self, z_parameters, v, tmax,  z_hyperparameters=None, check=True):
        def minus_rate_of_t(t):
            zt_parameters_numpy, vt = self.EllipticDynamics(t, z_parameters, v)
            if self.gibbs:
                zt_parameters = self.numpy_to_dict_of_tensors(zt_parameters_numpy, self.parameter_list)
                zt = self.merge_param_hyper(zt_parameters, z_hyperparameters)
            else:
                zt = self.numpy_to_dict_of_tensors(zt_parameters_numpy, self.key_of_z)
            z_grads_new, potential_energy_new = potential_grad(self.potential_fn, zt)
            if self.gibbs:
                z_parameters_grads_new, z_hyperparameters_grads_new = self.split_param_hyper(z_grads_new)
                z_parameters_grads_new_numpy = self.dict_of_tensors_to_numpy(z_parameters_grads_new)
            else:
                z_parameters_grads_new_numpy = self.dict_of_tensors_to_numpy(z_grads_new)
            gradU = z_parameters_grads_new_numpy - np.dot(np.linalg.inv(self.Sigma), (zt_parameters_numpy - self.z_ref))
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

    def EllipticDynamics(self, t, y0, w0):
        # simulate dy/dt = w, dw/dt = -y
        y_new = y0 * np.cos(t) + w0 * np.sin(t)
        w_new = -y0 * np.sin(t) + w0 * np.cos(t)
        return (y_new, w_new)

    def metropolis_step(self, current_betas, current_lambdas, current_tau):
        lambda_proposal = pyro.sample('prop_lambda', dist.HalfCauchy(1).expand([self.dim]))
        tau_proposal = pyro.sample('prop_tau', dist.HalfCauchy(1))
        size = len(current_betas)
        unif = torch.rand(size)
        lambdas = current_lambdas
        tau = current_tau
        for i in range(size):
            gauss_current = torch.distributions.normal.Normal(0, current_lambdas[i] * current_tau)
            gauss_prop = torch.distributions.normal.Normal(0, lambda_proposal[i] * current_tau)
            if torch.log(unif[i]) < (gauss_prop.log_prob(current_betas[i]) - gauss_current.log_prob(current_betas[i])):
                lambdas[i] = lambda_proposal[i]
                self._no_accepted_gibbs = self._no_accepted_gibbs + 1/(size+1) #CHANGE: ONLY BECAUSE TAU HAS SIZE 1
        current_sum = 0
        proposal_sum = 0
        for i in range(size):
            current_sum = current_sum + torch.distributions.normal.Normal(0, current_lambdas[i] * current_tau).log_prob(
                current_betas[i])
            proposal_sum = proposal_sum + torch.distributions.normal.Normal(0,
                                                                            current_lambdas[i] * tau_proposal).log_prob(
                current_betas[i])
        if torch.log(torch.rand(1)) < proposal_sum - current_sum:
            tau = current_tau
            self._no_accepted_gibbs = self._no_accepted_gibbs + 1 / (size + 1)  # CHANGE: ONLY BECAUSE TAU HAS SIZE 1
        return lambdas, tau

    def split_param_hyper(self, z):
        z_parameters = {self.parameter_list[0]: z[self.parameter_list[0]]}
        for i in range(1, len(self.parameter_list)):
            z_parameters.update({self.parameter_list[i]: z[self.parameter_list[i]]})
        z_hyperparameters = {self.hyperparameter_list[0]: z[self.hyperparameter_list[0]]}
        for i in range(1, len(self.hyperparameter_list)):
            z_hyperparameters.update({self.hyperparameter_list[i]: z[self.hyperparameter_list[i]]})

        return z_parameters, z_hyperparameters

    def merge_param_hyper(self, z_parameter, z_hyperparameter):
        z = z_parameter.copy()
        z.update(z_hyperparameter)
        return z

    def metropolis_step_RW(self, current_betas, current_lambdas, current_tau, scale):
        size = len(current_betas)
        current_a = torch.log(current_lambdas)
        a_proposal = current_a + dist.Normal(0, scale).sample(sample_shape=torch.Size([size]))
        tau_proposal = pyro.sample('prop_tau', dist.HalfCauchy(1))
        unif = torch.rand(size)
        lambdas = current_lambdas
        tau = current_tau
        for i in range(size):
            gauss_current = torch.distributions.normal.Normal(0, current_lambdas[i] * current_tau)
            gauss_prop = torch.distributions.normal.Normal(0, torch.exp(a_proposal[i]) * current_tau)
            half_cauchy = torch.distributions.half_cauchy.HalfCauchy(1)
            if torch.log(unif[i]) < (gauss_prop.log_prob(current_betas[i])+a_proposal[i]+half_cauchy.log_prob(torch.exp(a_proposal[i])) - gauss_current.log_prob(current_betas[i])-current_a[i]-half_cauchy.log_prob(torch.exp(current_a[i]))):
                lambdas[i] = torch.exp(a_proposal[i])
                self._no_accepted_gibbs = self._no_accepted_gibbs + 1 / (
                            size + 1)  # CHANGE: ONLY BECAUSE TAU HAS SIZE 1
        current_sum = 0
        proposal_sum = 0
        for i in range(size):
            current_sum = current_sum + torch.distributions.normal.Normal(0, current_lambdas[i] * current_tau).log_prob(
                current_betas[i])
            proposal_sum = proposal_sum + torch.distributions.normal.Normal(0,
                                                                            current_lambdas[i] * tau_proposal).log_prob(
                current_betas[i])
        if torch.log(torch.rand(1)) < proposal_sum - current_sum:
            tau = current_tau
            self._no_accepted_gibbs = self._no_accepted_gibbs + 1 / (size + 1)  # CHANGE: ONLY BECAUSE TAU HAS SIZE 1
        return lambdas, tau

    def rwmh_HP(self, current_parametres, current_hyperparametres, scale):
        lambdas = self.find_keys_with(["lambda"],current_hyperparametres)
        taus = self.find_keys_with(["tau"], current_hyperparametres)


        for type_of_lambda in lambdas:
            layer = self.extract_layer(type_of_lambda) #extract the layer to which the hyperparametres belong
            type_of_parameter = self.extract_type_of_parametre(type_of_lambda) #extract the type of parametre to which the hyperparametre belongs
            corresponding_parameters_key = self.find_keys_with([layer] + [type_of_parameter], current_parametres)
            corresponding_tau_key = self.find_keys_with([layer] + [type_of_parameter] + ["tau"], current_hyperparametres)

            """
            print("looking for: ", [layer] + [type_of_parameter], "\n")
            print("in :", list(current_parametres.keys()), "\n")
            print("I found: ", corresponding_parameters_key,"\n")
            """
            corresponding_parameters = current_parametres[corresponding_parameters_key[0]].squeeze()
            corresponding_tau = current_hyperparametres[corresponding_tau_key[0]]
            this_lambda = current_hyperparametres[type_of_lambda].squeeze()
            current_a = torch.log(this_lambda)
            if this_lambda.dim() == 0:
                this_lambda = this_lambda.unsqueeze(0)
                corresponding_parameters = corresponding_parameters.unsqueeze(0)
                current_a = current_a.unsqueeze(0)
            size = len(this_lambda)
            a_proposal = current_a + dist.Normal(0, scale).sample(sample_shape=torch.Size([size]))

            unif = torch.rand(size)



            for i in range(size):
                gauss_current = torch.distributions.normal.Normal(0, this_lambda[i] * corresponding_tau)
                gauss_prop = torch.distributions.normal.Normal(0, torch.exp(a_proposal[i]) * corresponding_tau)
                half_cauchy = torch.distributions.half_cauchy.HalfCauchy(1)
                #print((gauss_prop.log_prob(corresponding_parameters[i])+a_proposal[i]+half_cauchy.log_prob(torch.exp(a_proposal[i])) -
                                         #gauss_current.log_prob(corresponding_parameters[i])-current_a[i]-half_cauchy.log_prob(torch.exp(current_a[i]))))
                if torch.log(unif[i]) < (gauss_prop.log_prob(corresponding_parameters[i])+a_proposal[i]+half_cauchy.log_prob(torch.exp(a_proposal[i])) -
                                         gauss_current.log_prob(corresponding_parameters[i])-current_a[i]-half_cauchy.log_prob(torch.exp(current_a[i]))):
                    this_lambda[i] = torch.exp(a_proposal[i])
                    self._no_accepted_gibbs = self._no_accepted_gibbs + 1 / (3*100 + 5)  # CHANGE: ONLY BECAUSE TAU HAS SIZE 1

            #Give the correct shape again to the new hyperparametres
            current_hyperparametres[type_of_lambda] = this_lambda.reshape(self.shapes_hyperparametres[type_of_lambda])


        for type_of_tau in taus:
            layer = self.extract_layer(type_of_tau) #extract the layer to which the hyperparametres belong
            type_of_parameter = self.extract_type_of_parametre(type_of_tau) #extract the type of parametre to which the hyperparametre belongs
            corresponding_parameters_key = self.find_keys_with([layer] + [type_of_parameter], current_parametres)
            corresponding_lambdas_key = self.find_keys_with([layer] + [type_of_parameter] + ["lambda"], current_hyperparametres)
            corresponding_parameters = current_parametres[corresponding_parameters_key[0]].squeeze()
            corresponding_lambdas = current_hyperparametres[corresponding_lambdas_key[0]].squeeze()

            if corresponding_lambdas.dim() == 0:
                corresponding_lambdas = this_lambda.unsqueeze(0)
                corresponding_parameters = corresponding_parameters.unsqueeze(0)

            size = len(corresponding_lambdas)

            this_tau = current_hyperparametres[type_of_tau]
            a_current = torch.log(this_tau)
            a_proposal = a_current + dist.Normal(0, scale).sample()
            current_sum = 0
            proposal_sum = 0
            for i in range(size):
                current_sum = current_sum + torch.distributions.normal.Normal(0, corresponding_lambdas[i] * this_tau).log_prob(
                    corresponding_parameters[i])
                proposal_sum = proposal_sum + torch.distributions.normal.Normal(0,
                                                                                corresponding_lambdas[i] * torch.exp(a_proposal)).log_prob(
                    corresponding_parameters[i])
            if torch.log(torch.rand(1)) < proposal_sum - current_sum + (a_proposal - a_current):
                current_hyperparametres[type_of_tau] = torch.exp(a_proposal)
                self._no_accepted_gibbs = self._no_accepted_gibbs + 1 / (3*100 + 5)   # CHANGE: ONLY BECAUSE TAU HAS SIZE 1

        return current_hyperparametres

    def subsample(self):
        indices = torch.randint(self._args[0].shape[0], (self.batch_size,))
        x_train_subsample = self._args[0][indices, :]
        y_train = self._args[1][indices]
        return x_train_subsample, y_train
    def find_keys_with(self, words, dictionary):
        """
        :param words: a LIST of words to find among the keys of dictionary.
        :param dictionary: a DICTionary in which to find for specified words.
        :return: a LIST of keys from "dictionary" that contain ALL the strings in words.
        """
        keys = []
        for key in list(dictionary.keys()):
            check = 0
            for word in words:
                if key.find(word) > -1:
                    check +=1
            if check == len(words):
                keys.append(key)
        return keys

    def extract_layer(self, key):
        layer_found = None
        for layer in self.layers:
            if key.find(layer) > -1:
                layer_found = layer
        return layer_found

    def extract_type_of_parametre(self, key):
        type_found = None
        for type in self.types_of_parameters:
            if key.find(type) > -1:
                type_found = type
        return type_found

    def rate_of_t_gibbs(self, z_parameters, z_hyperparameters, v, t):
        zt_parameters_numpy, vt = self.EllipticDynamics(t, z_parameters, v)
        zt_parameters = self.numpy_to_dict_of_tensors(zt_parameters_numpy, self.parameter_list)
        zt = self.merge_param_hyper(zt_parameters, z_hyperparameters)
        z_grads_new, potential_energy_new = potential_grad(self.potential_fn, zt)
        z_parameters_grads_new, z_hyperparameters_grads_new = self.split_param_hyper(z_grads_new)
        z_parameters_grads_new_numpy = self.dict_of_tensors_to_numpy(z_parameters_grads_new)
        gradU = z_parameters_grads_new_numpy - np.dot(np.linalg.inv(self.Sigma), (zt_parameters_numpy - self.z_ref))
        ft = np.dot(vt, gradU)
        return np.maximum(0, ft)

    def rate_of_t_nongibbs(self, z, v, lista, t):
        zt_numpy, vt = self.EllipticDynamics(t, z, v)
        zt = self.numpy_to_dict_of_tensors(zt_numpy, lista)
        #zt = self.merge_param_hyper(zt, z)
        z_grads_new, potential_energy_new = potential_grad(self.potential_fn, zt)
        #z_parameters_grads_new, z_hyperparameters_grads_new = self.split_param_hyper(z_grads_new)
        z_grads_new_numpy = self.dict_of_tensors_to_numpy(z_grads_new)
        gradU = z_grads_new_numpy - np.dot(np.linalg.inv(18), (zt_numpy - self.z_ref))
        ft = np.dot(vt, gradU)
        return np.maximum(0, ft)