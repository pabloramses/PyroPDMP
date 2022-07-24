import numpy as np
from boomerangJULIA import EllipticDynamics, switchingtime

def Boomerang_gibbs(g_E, M1, T, x_ref, Sigma_inv, refresh_rate=1.0):
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
    condition = np.random.normal(Mu_comp[2] + (Sigma_comp[2, 1] / Sigma_comp[1, 1]) * (x[1] - Mu_comp[1]),
                                 Sigma_comp[2, 2] - ((Sigma_comp[2, 1] * Sigma_comp[1, 2]) / Sigma_comp[1, 1]), 1)
    gradU = g_E(x, condition[0]) - np.dot(Sigma_inv, (x-x_ref)) # O(d^2) to compute
    gradBound = g_E(x_ref, 1000)  # O(d^2) to compute
    M2 = np.sqrt(np.dot(gradBound, gradBound))
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
    a = np.dot(v, gradU) #starting point for the rate
    b = M1 * phaseSpaceNorm**2 + M2 * phaseSpaceNorm #slope term for the rate

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
        a = a + b*dt  #rate at the end of dt period
        gradU = g_E(x, condition[0]) - np.dot(Sigma_inv, (x-x_ref)) # O(d^2) to compute, gradient at the end of dt period

        if not finished and dt_switch_proposed < dt_refresh:
            switch_rate = np.dot(v, gradU) # no need to take positive part, true rate at the end of dt period
            simulated_rate = a #bound at the end of dt period
            if simulated_rate < switch_rate:
                print("simulated rate: ", simulated_rate)
                print("actual switching rate: ", switch_rate)
                print("switching rate exceeds bound.")
                print(dt)

            #simul3 = 0.01
            if np.random.rand() * simulated_rate <= switch_rate:
                # obtain new velocity by reflection
                skewed_grad = np.dot(np.transpose(Sigma_sqrt), gradU)
                v = v - 2 * (switch_rate / np.dot(skewed_grad,skewed_grad)) * np.dot(Sigma_sqrt, skewed_grad)
                phaseSpaceNorm = np.sqrt(np.dot(x-x_ref,x-x_ref) + np.dot(v,v))
                a = -switch_rate #locates the bound for next step
                b = M1 * phaseSpaceNorm**2 + M2 * phaseSpaceNorm
                updateSkeleton = True
                accepted_switches += 1
            else:
                a = switch_rate
                updateSkeleton = False
                rejected_switches += 1

            # update refreshment time and switching time bound
            dt_refresh = dt_refresh - dt_switch_proposed

        if (not finished and dt_switch_proposed >= dt_refresh):
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
            condition = np.random.normal(Mu_comp[2] + (Sigma_comp[2,1] / Sigma_comp[1,1]) * (x[1] - Mu_comp[1]),Sigma_comp[2,2]-((Sigma_comp[2,1] * Sigma_comp[1,2])/Sigma_comp[1,1]),1)
            updateSkeleton = False

    return (t_skeleton, x_skeleton, v_skeleton, x_ref)


Mu = np.array([0,1])
Sigma = np.array([[1,0],[0,1]])
Sigma_inv = np.linalg.inv(Sigma)


Sigma_comp = np.array([[1,0,0],[0,1,0.2],[0,0.2,1]])
Mu_comp = np.array([0,1,1])
Sigma_cond = np.array([[Sigma_comp[0,0],0],[0,Sigma_comp[1,1]-((Sigma_comp[1,2] * Sigma_comp[2,1])/Sigma_comp[2,2])]])
Sigma_cond_inv = np.linalg.inv(Sigma_cond)
M1 = np.linalg.norm(Sigma_cond_inv)
def boomerang_gradient(x, condition):
    Mu_cond = np.array([Mu_comp[0], Mu_comp[1] + (Sigma_comp[1,2] / Sigma_comp[2,2]) * (condition - Mu_comp[2])])
    Sigma_cond = np.array([[Sigma_comp[0,0],0],[0,Sigma_comp[1,1]-((Sigma_comp[1,2] * Sigma_comp[2,1])/Sigma_comp[2,2])]])
    Sigma_cond_inv = np.linalg.inv(Sigma_cond)
    return np.dot(x-Mu_cond, Sigma_cond_inv)

x_ref = np.array([0,0])
Sigma_inv_ref = Sigma_inv

t, x, v, ref = Boomerang_gibbs(boomerang_gradient, M1, 10000, x_ref, Sigma_inv_ref, refresh_rate=1.0)