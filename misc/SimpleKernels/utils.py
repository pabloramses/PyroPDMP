import numpy as np

class nhppSample:
    def __init__(self, dim, inv_Lambda=None, Lambda=None, tol=10 ** -2):
        self.inv_Lambda = inv_Lambda
        self.Lambda = Lambda
        self.tol = tol
        self.dim = dim
    def cinlar(self, x=None, v=None):
        """
        Outputs event times from a non-homogeneous Poisson Process for a known integrated intensity function, although its inverse may or may not be known in which case
        its values are approximated.
        :param length: dimension
        :param inv_Lambda (optional): inverse of integrated intensity
        :param Lambda (optional): has to be given if inv_Lambda is not given.
        :param tol (optional): controls the approximation error.
        :return:
        """
        expTimes = np.random.exponential(1, self.dim)
        if self.inv_Lambda:
            self.eventTimes = self.inv_Lambda(expTimes, x, v)
        else:
            self.eventTimes = []
            for i in range(self.dim):
                t = 0
                while np.abs(self.Lambda(t) - expTimes[i]) > self.tol:
                    t = t + 0.01
                self.eventTimes.append(t)
        self.eventTimes = np.array([self.eventTimes])
        return self.eventTimes
    def thin(self):
        pass




