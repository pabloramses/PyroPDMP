import numpy as np


def cinlar(length, inv_Lambda=None, Lambda=None, tol=10 ** -2):
    """
    Outputs event times from a non-homogeneous Poisson Process for a known integrated intensity function, although its inverse may or may not be known in which case
    its values are approximated.
    :param length: number of
    :param inv_Lambda (optional): inverse of integrated intensity
    :param Lambda (optional): has to be given if inv_Lambda is not given.
    :param tol (optional): controls the approximation error.
    :return:
    """
    expTimes = np.random.exponential(1, lenght)
    if inv_Lambda:
        eventTimes = inv_Lambda(expTimes)
    else:
        eventTimes = []
        for i in range(length):
            t = 0
            while np.abs(t - expTimes[i]) > tol:
                t = t + 0.01
            eventTimes.append(t)

    return eventTimes


def gradient(a):
    """
    :return:
    d-dimensional array
    """
    pass


def rate_ZZ(time, pos, vel):
    rate = (-vel * gradient(pos + time * vel) > 0) * (-vel * gradient(pos + time * vel))
    return rate

def zzSampler(gradient, rate, bound, initial_pos, niter):
    times = [0]
    d = len(initial_pos)
    pos = np.zeros((d, niter))
    vel = np.zeros((d, niter))

    pos[:,0] = initial_pos
    vel[:,0] = -1+2*np.random.randint(2, size=d)

    for i in range(niter):
        upperTime = cinlar(d, inv_Lambda=bound)


        t = upperTime[np.argmin(upperTime)]
        times.append(t)
        pos[:,i] = pos[:,i]+t*vel[:,i]

        r = np.float32(rate(t, pos[i+1], vel[i])[k])
        b = np.float32(bound(t))
        u = np.random.random()
        vel[:, i + 1] = vel
        if u<(r/b):
            vel[i, i + 1] = (-1) * vel[i, i + 1]

    return [Time, Position, Velocity]


