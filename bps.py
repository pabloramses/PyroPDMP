import numpy as np
from utils import nhppSample

class bps(nhppSample):
    def __init__(self, gradient, bound, initial_pos, niter, lr, int_bound = None, inv_int_bound = None, tol=10 ** -2):
        super(bps, self).__init__(1, inv_Lambda=inv_int_bound, Lambda=int_bound, tol=tol)
        self.d = len(initial_pos)
        self.gradient = gradient
        self.bound = bound
        self.initial_pos = initial_pos
        self.niter = niter
        self.lr = lr


    def rate(self, time, pos, vel):
        self.Rate = (np.dot(vel, self.gradient(pos + time * vel))>0)*np.dot(vel, self.gradient(pos + time * vel))

    def sample(self):
        self.times = [0]
        d = len(self.initial_pos)
        self.pos = np.zeros((self.d, self.niter))
        self.vel = np.zeros((self.d, self.niter))
        self.pos[:, 0] = self.initial_pos
        self.vel[:, 0] = np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d), size=1)

        for i in range(1,self.niter):
            refresh = np.random.exponential(self.lr)

            accepted = True

            t = 0
            while accepted:
                prop_x = self.pos[:, i - 1]
                prop_v = self.vel[:, i - 1]

                if self.inv_Lambda:
                    upperTime = super().cinlar(x=prop_x, v=prop_v)
                else:
                    upperTime = super().cinlar()
                t = t + upperTime[0]
                if t>refresh:
                    self.pos[:, i] = self.pos[:, i-1] + refresh * self.vel[:,i-1]
                    self.vel[:, i] = np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d))
                    self.times.append(refresh)
                    break

                prop_x = prop_x+t*prop_v

                self.rate(t, self.pos[:, i - 1], self.vel[:, i - 1])

                r = np.float32(self.Rate)
                b = np.float32(self.bound(t, 0, self.pos[:, i - 1], self.vel[:, i - 1]))
                u = np.random.random()
                if u < (r/b):
                    self.pos[:, i] = prop_x
                    self.vel[:, i] = prop_v - 2*np.dot(((np.dot(prop_v,self.gradient(prop_x)))/(np.dot(self.gradient(prop_x),self.gradient(prop_x)))),self.gradient(prop_x))
                    self.times.append(t)
                    accepted = False
