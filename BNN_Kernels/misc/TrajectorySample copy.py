import numpy as np
import torch

class TrajectorySample():
    def __init__(self, kernel, skeleton, n_samples, ellipticQ=True, parameter_list = None, hyperparameter_list = None):
        self.parameter_list = parameter_list
        self.hyperparameter_list = hyperparameter_list
        self.x_skeleton_dict = skeleton
        self.extract_keys_of_z(self.x_skeleton_dict)
        self.n = self.x_skeleton_dict[self.key_of_z[0]].shape[0]
        if not hasattr(kernel, 'v_skeleton'):
            raise ValueError("The mcmc still needs to be run")
        else:
            self.v_skeleton = kernel.v_skeleton
            self.t_skeleton = np.cumsum(kernel.dts[-self.n:]) - kernel.dts[-self.n]

        self.n_samples = n_samples
        self.ellipticQ = ellipticQ
        self.x_ref = kernel.z_ref

        self.dimension_of_components(self.x_skeleton_dict)
        self.shapes_of_components(self.x_skeleton_dict)
        self.ExtractSamples()

    def extract_keys_of_z(self, z):
        if len(list(z.keys())) == 1:
            self.key_of_z = list(z.keys())[0]
        else:
            self.key_of_z = list(z.keys())

    def shapes_of_components(self, z):
        self.shapes = {}
        for value, key in enumerate(z):
            self.shapes.update({key: z[key][0].shape})


    def EllipticDynamics(self, t, y0, w0):
        # simulate dy/dt = w, dw/dt = -y
        y_new = y0 * np.cos(t) + w0 * np.sin(t)
        w_new = -y0 * np.sin(t) + w0 * np.cos(t)
        return (y_new, w_new)

    def dict_of_tensors_to_numpy_V(self, z):
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
            for t in range(1, z[key_of_z[0]].shape[0]):
                z_numpy_t = z[key_of_z[0]][t].numpy()
                for j in range(1, len(key_of_z)):
                    # convert to Numpy array
                    z_numpy_j = z[key_of_z[j]][t].numpy()
                    z_numpy_t = np.append(z_numpy_t, z_numpy_j)
                z_numpy = np.vstack((z_numpy, z_numpy_t))
        return z_numpy

    def dimension_of_components(self, z):
        if type(self.key_of_z) == str:
            self.dimensions = [1]
        else:
            self.dimensions = [len(z[self.key_of_z[0]][0])]
            for j in range(1, len(self.key_of_z)):
                if z[self.key_of_z[j]][0].dim() == 0:
                    self.dimensions.append(1)
                else:
                    self.dimensions.append(torch.prod(
                        torch.tensor(z[self.key_of_z[j]][0].shape)).item())  # very rustic way to get the dimensions
        return self.dimensions



    def numpy_to_dict_of_tensors_V(self, z_numpy):
        z_numpy = np.float32(z_numpy)
        limits = np.cumsum(self.dimensions)

        if type(self.key_of_z) == str:
            z = {self.key_of_z: torch.from_numpy(z_numpy)}
        else:
            z = {self.key_of_z[0]: torch.from_numpy(z_numpy[0,0:limits[0]]).reshape(self.shapes[self.key_of_z[0]])}
            for j in range(1, len(self.dimensions)):
                # convert to Numpy array
                z.update({self.key_of_z[j]: torch.from_numpy(z_numpy[0,limits[j - 1]: limits[j]]).reshape(
                    self.shapes[self.key_of_z[j]])})
            for t in range(1, self.n_samples):
                z[self.key_of_z[0]] = torch.vstack((z[self.key_of_z[0]],torch.from_numpy(z_numpy[t, 0:limits[0]]).reshape(self.shapes[self.key_of_z[0]])))
                for j in range(1, len(self.dimensions)):
                    # convert to Numpy array
                    z[self.key_of_z[j]] = torch.vstack((z[self.key_of_z[j]],torch.from_numpy(z_numpy[t, limits[j - 1]: limits[j]]).reshape(self.shapes[self.key_of_z[j]])))

        for i in self.key_of_z:
            correct_shape = self.x_skeleton_dict[i].shape
            tensor_correct_shape = torch.tensor(correct_shape)
            tensor_correct_shape[0] = self.n_samples
            correct_shape = torch.Size(tensor_correct_shape)
            z[i] = z[i].reshape(correct_shape)
        return z

    def ExtractSamples(self):
        x_skeleton_parameters, x_skeleton_hyperparameters = self.split_param_hyper(self.x_skeleton_dict)
        x_skeleton = self.dict_of_tensors_to_numpy_V(x_skeleton_parameters)
        T = self.t_skeleton[-1]
        dt = T / (self.n_samples - 1)
        t = 0.0
        v = self.v_skeleton[0]
        x = x_skeleton[0]
        self.sample_numpy = x
        counter = 0
        for i in range(1, self.n_samples):
            t_max = (i - 1) * dt
            while (counter + 1 < len(self.t_skeleton) and self.t_skeleton[counter + 1] < t_max):
                counter = counter + 1
                x = x_skeleton[counter, 0:len(v)]
                v = self.v_skeleton[counter]
                t = self.t_skeleton[counter]
            if (self.ellipticQ):
                (y, v) = self.EllipticDynamics(t_max - t, x - self.x_ref, v)
                x = y + self.x_ref
            else:
                x = x + v * (t_max - t)
            t = t_max
            self.sample_numpy = np.vstack((self.sample_numpy, x))
        self.sample = self.numpy_to_dict_of_tensors_V(self.sample_numpy)
        return self.sample_numpy

    def split_param_hyper(self, z):
        z_parameters = {self.parameter_list[0]: z[self.parameter_list[0]]}
        for i in range(1, len(self.parameter_list)):
            z_parameters = {self.parameter_list[i]: z[self.parameter_list[i]]}
        z_hyperparameters = {self.hyperparameter_list[0]: z[self.hyperparameter_list[0]]}
        for i in range(1, len(self.hyperparameter_list)):
            z_hyperparameters.update({self.hyperparameter_list[i]: z[self.hyperparameter_list[i]]})

        return z_parameters, z_hyperparameters

    def merge_param_hyper(self, z_parameter, z_hyperparameter):
        z = z_parameter.copy()
        z.update(z_hyperparameter)
        return z