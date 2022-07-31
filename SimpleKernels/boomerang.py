import numpy as np
from utils import nhppSample



class boomerang(nhppSample):
    def __init__(self, sigma_ref, c, gradient, bound, initial_pos, niter, lr, int_bound = None, inv_int_bound = None, tol=10 ** -2):
        super(boomerang, self).__init__(1, inv_Lambda=inv_int_bound, Lambda=int_bound, tol=tol)
        self.d = len(initial_pos)
        self.gradient = gradient
        self.bound = bound
        self.initial_pos = initial_pos
        self.niter = niter
        self.lr = lr
        self.sigma_ref = sigma_ref
        self.c = c

    def elliptic_dynamics(self,x, c, v, t):
        x_t = c + (x - c) * np.cos(t) + v * np.sin(t)
        v_t = -(x - c) * np.sin(t) + v * np.cos(t)
        return np.array([x_t, v_t])

    def rate(self, time, pos, vel):
        #skel = self.elliptic_dynamics(pos, self.c, vel, time)
        self.Rate = (np.dot(vel, self.gradient(pos))>0)*np.dot(vel, self.gradient(pos))

    def sample(self):
        self.times = [0]
        d = len(self.initial_pos)
        self.pos = np.zeros((self.d, self.niter))
        self.vel = np.zeros((self.d, self.niter))

        self.pos[:, 0] = self.initial_pos
        self.vel[:, 0] = np.random.multivariate_normal(np.zeros(self.d), self.sigma_ref, size=1)

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
                    upperTime = super().cinlar(x=prop_x, v=prop_v)
                t = t + upperTime[0]
                if t>refresh:
                    skel = self.elliptic_dynamics(self.pos[:, i-1], self.c, self.vel[:,i-1], refresh)
                    self.pos[:, i] = skel[0]
                    self.vel[:, i] = np.random.multivariate_normal(np.zeros(self.d), self.sigma_ref)
                    self.times.append(refresh)
                    break

                skel = self.elliptic_dynamics(self.pos[:, i - 1], self.c, self.vel[:, i - 1], t)
                prop_x = skel[0]
                prop_v = skel[1]

                self.rate(t, self.pos[:, i - 1], self.vel[:, i - 1])

                r = np.float32(self.Rate)
                b = np.float32(self.bound(t, 0, self.pos[:, i - 1], self.vel[:, i - 1]))
                u = np.random.random()
                if u < (r/b):
                    self.pos[:, i] = prop_x
                    self.vel[:, i] = prop_v - 2*np.dot(((np.dot(prop_v,self.gradient(prop_x)))/(np.linalg.norm(np.dot(np.linalg.cholesky(self.sigma_ref),self.gradient(prop_x)))**2)),np.dot(self.sigma_ref,self.gradient(prop_x)))
                    self.times.append(t)
                    accepted = False