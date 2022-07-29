import numpy as np

def switchingtime(a, b, u = None):
# generate switching time for rate of the form max(0, a + b s) + c
# under the assumptions that b > 0, c > 0
    if u:
        pass
    else:
        u = np.random.rand()
    if (b > 0):
        if (a < 0):
            return -a/b + switchingtime(0.0, b, u)
        else: # a >= 0
            return -a/b + np.sqrt(a**2/b**2 - 2 * np.log(u)/b)

    elif (b == 0): # degenerate case
        if (a < 0):
            return np.inf
        else: # a >= 0
            return -np.log(u)/a

    else: # b <= 0
        if (a <= 0):
            return np.inf
        else: # a > 0
            y = -np.log(u); t1=-a/b
            if (y >= a * t1 + b *(t1**2)/2):
                return np.inf
            else:
                return -a/b - np.sqrt((a**2)/(b**2) + 2 * y /b)

def EllipticDynamics(t, y0, w0):
    # simulate dy/dt = w, dw/dt = -y

    y_new = y0 * np.cos(t) + w0 * np.sin(t)
    w_new = -y0 * np.sin(t) + w0 * np.cos(t)

    return (y_new, w_new)



def Boomerang(g_E, M1, T, x_ref, Sigma_inv, refresh_rate=1.0):
    # g_E! and h_E! are gradient and hessian of negative log density E respectively
    #  (implemented to be compatible with the Optim package, so g_E!(storage,x), h_E!(storage,x))
    # M1 is a constant such that |∇^2 E(x)| <= M1  for all x.
    # M2 is a constant such that |∇E(x_ref)| ≤ M2
    # Boomerang is implemented with affine bound.



    Sigma = np.linalg.inv(Sigma_inv)

    Sigma_sqrt = np.linalg.cholesky(Sigma)



    dim = len(x_ref)

    # gradU(x) = g_E!(dummy,x) - Σ_inv * (x-x_ref); # O(d^2) to compute

    t = 0.0
    #simul1 = np.array([1.02, -2.02])
    x = x_ref
    v = np.dot(Sigma_sqrt, np.random.normal(0,1,dim))
    #v = np.dot(Sigma_sqrt, simul1)
    gradU = g_E(x) - np.dot(Sigma_inv, (x-x_ref)) # O(d^2) to compute
    M2 = np.sqrt(np.dot(gradU,gradU))
    updateSkeleton = False
    finished = False
    x_skeleton = x
    v_skeleton = v
    t_skeleton = np.array([t])

    #simul2 = 0.34
    dt_refresh = -np.log(np.random.rand())/refresh_rate
    #dt_refresh = -np.log(simul2) / refresh_rate

    rejected_switches = 0
    accepted_switches = 0
    phaseSpaceNorm = np.sqrt(np.dot(x-x_ref,x-x_ref) + np.dot(v,v))
    a = np.dot(v, gradU)
    b = M1 * phaseSpaceNorm**2 + M2 * phaseSpaceNorm

    while not finished :
        dt_switch_proposed = switchingtime(a,b)
        dt = np.minimum(dt_switch_proposed,dt_refresh)
        if t + dt > T:
            dt = T - t
            finished = True
            updateSkeleton = True

        (y, v) = EllipticDynamics(dt, x-x_ref, v)
        x = y + x_ref
        t = t + dt
        a = a + b*dt
        gradU = g_E(x) - np.dot(Sigma_inv, (x-x_ref)) # O(d^2) to compute
        if not finished and dt_switch_proposed < dt_refresh:
            switch_rate = np.dot(v, gradU) # no need to take positive part
            simulated_rate = a
            if simulated_rate < switch_rate:
                print("simulated rate: ", simulated_rate)
                print("actual switching rate: ", switch_rate)
                print("switching rate exceeds bound.")

            #simul3 = 0.01
            if np.random.rand() * simulated_rate <= switch_rate:
                # obtain new velocity by reflection
                skewed_grad = np.dot(np.transpose(Sigma_sqrt), gradU)
                v = v - 2 * (switch_rate / np.dot(skewed_grad,skewed_grad)) * np.dot(Sigma_sqrt, skewed_grad)
                phaseSpaceNorm = np.sqrt(np.dot(x-x_ref,x-x_ref) + np.dot(v,v))
                a = -switch_rate
                b = M1 * phaseSpaceNorm**2 + M2 * phaseSpaceNorm
                updateSkeleton = True
                accepted_switches += 1
            else:
                a = switch_rate
                updateSkeleton = False
                rejected_switches += 1

            # update refreshment time and switching time bound
            dt_refresh = dt_refresh - dt_switch_proposed

        elif not finished:
            # so we refresh
            updateSkeleton = True
            v = np.dot(Sigma_sqrt, np.random.normal(0,1,dim))
            phaseSpaceNorm = np.sqrt(np.dot(x-x_ref,x-x_ref) + np.dot(v,v))
            a = np.dot(v, gradU)
            b = M1 * phaseSpaceNorm**2 + M2 * phaseSpaceNorm

            # compute new refreshment time
            #simul4 = 0.23
            #dt_refresh = -np.log(simul4) / refresh_rate
            dt_refresh = -np.log(np.random.rand())/refresh_rate

        else:
            pass

        if updateSkeleton:
            x_skeleton = np.vstack((x_skeleton, np.transpose(x)))
            v_skeleton = np.vstack((v_skeleton, np.transpose(v)))
            t_skeleton = np.append(t_skeleton, t)
            updateSkeleton = False

    print("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    print("number of proposed switches: ", accepted_switches + rejected_switches)
    #print(rejected_switches)
    return (t_skeleton, x_skeleton, v_skeleton, x_ref)

def boomerang_gradient(x):
    grad = np.dot(x-Mu, Sigma_inv)
    return grad

"""TARGET VALUES
Sigma = np.array([[3,0.5],[0.5,3]])
Sigma_inv = np.linalg.inv(Sigma)
Mu = np.array([40,60])

"REFERENCE VALUES"
x_ref = np.zeros(2)
Sigma_inv_ref = Sigma_inv
M1 = np.linalg.norm(Sigma_inv)

t, x, v, ref = Boomerang(boomerang_gradient, M1, 10000, x_ref, Sigma_inv_ref, refresh_rate=1)
import matplotlib.pyplot as plt
plt.plot(x[:,0], x[:,1])
np.mean(x[:,0])
np.cov(x[:,0], x[:,1])"""
