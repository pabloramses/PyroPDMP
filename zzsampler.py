import numpy as np


def gradient(a):
    """
    :return:
    d-dimensional array
    """
    pass

def bound():

def zzSampler(gradient, rate, bound, initial_pos, niter, inv_bound=None, int_bound=None):
    times = [0]
    d = len(initial_pos)
    pos = np.zeros((d, niter))
    vel = np.zeros((d, niter))

    pos[:,0] = initial_pos
    vel[:,0] = -1+2*np.random.randint(2, size=d)

    for i in range(niter):
        if inv_bound:
            upperTime = cinlar(d, inv_Lambda=inv_bound)
        else:
            upperTime = cinlar(d, Lambda = int_bound)

        t = upperTime[np.argmin(upperTime)]
        times.append(t)
        pos[:,i] = pos[:,i]+t*vel[:,i]

        r = np.float32(rate(t, pos[i+1], vel[i])[k])
        b = np.float32(bound(t)) #will depend on both state and velocity
        u = np.random.random()
        vel[:, i + 1] = vel
        if u<(r/b):
            vel[i, i + 1] = (-1) * vel[i, i + 1]

    return [Time, Position, Velocity]