import numpy as np
from utils import nhppSample

class bps(nhppSample):
    def __init__(self, gradient, bound, initial_pos, niter, int_bound = None, inv_int_bound = None, tol=10 ** -2):
        super(bps, self).__init__(1, inv_Lambda=inv_int_bound, Lambda=int_bound, tol=tol)
        self.d = len(initial_pos)
        self.gradient = gradient
        self.bound = bound
        self.initial_pos = initial_pos
        self.niter = niter


    def rate(self, time, pos, vel):
        print("gradient", self.gradient(pos + time * vel))
        print("coeff", np.dot(vel, self.gradient(pos + time * vel)))
        self.Rate = (np.dot(vel, self.gradient(pos + time * vel))>0)*np.dot(vel, self.gradient(pos + time * vel))

    def sample(self):
        self.times = [0]
        d = len(self.initial_pos)
        self.pos = np.zeros((self.d, self.niter))
        self.vel = np.zeros((self.d, self.niter))
        print(self.pos.shape)
        print(self.initial_pos.shape)

        self.pos[:, 0] = self.initial_pos
        self.vel[:, 0] = np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d), size=1)

        for i in range(1,self.niter):
            if self.inv_Lambda:
                upperTime = super().cinlar(x=self.pos[:,i], v=self.vel[:,i])
            else:
                upperTime = super().cinlar()
            t = upperTime[0]
            print(t)
            self.times.append(t)
            self.pos[:, i] = self.pos[:, i-1]+t*self.vel[:, i-1]

            self.rate(t, self.pos[:, i], self.vel[:, i-1])

            r = np.float32(self.Rate)
            b = np.float32(self.bound(t, 0, self.pos[:, i], self.vel[:, i-1]))
            u = np.random.random()
            print("rate", r)
            print("bound",b)
            if u < (r/b):
                self.vel[:, i] = self.vel[:,i-1] - 2*np.dot(((np.dot(self.vel[:,i-1],self.gradient(self.pos[:, i])))/(np.dot(self.gradient(self.pos[:, i]),self.gradient(self.pos[:, i])))),self.gradient(self.pos[:, i]))
            else:
                self.vel[:, i] = np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d), size=1)
