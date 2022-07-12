import numpy as np
from utils import nhppSample

class zigzag(nhppSample):
    def __init__(self, gradient, bound, initial_pos, niter, int_bound = None, inv_int_bound = None, tol=10 ** -2):
        self.dim = len(initial_pos)
        super(zigzag, self).__init__(self.dim, inv_Lambda=inv_int_bound, Lambda=int_bound, tol=tol)
        self.gradient = gradient
        self.bound = bound
        self.initial_pos = initial_pos
        self.niter = niter


    def rate(self, time, pos, vel):
        self.Rate = (vel * self.gradient(pos + time * vel) > 0) * (vel * self.gradient(pos + time * vel))

    def sample(self):
        self.times = [0]
        d = len(self.initial_pos)
        self.pos = np.zeros((self.dim, self.niter))
        self.vel = np.zeros((self.dim, self.niter))

        self.pos[:, 0] = self.initial_pos
        self.vel[:, 0] = -1+2*np.random.randint(2, size=self.dim)

        for i in range(1,self.niter):
            if self.inv_Lambda:
                upperTime = super().cinlar(x=self.pos[:,i], v=self.vel[:,i])
            else:
                upperTime = super().cinlar()
            i0 = np.argmin(upperTime[0])
            t = upperTime[0][i0]

            self.times.append(t)
            self.pos[:, i] = self.pos[:, i-1]+t*self.vel[:, i-1]

            self.rate(t, self.pos[:, i], self.vel[:, i-1])

            r = np.float32(self.Rate[i0])
            b = np.float32(self.bound(t, i0, self.pos[:, i], self.vel[:, i-1]))
            u = np.random.random()
            self.vel[:, i] = self.vel[:,i-1]
            if u < (r/b):

                self.vel[i0, i] = (-1) * self.vel[i0, i-1]