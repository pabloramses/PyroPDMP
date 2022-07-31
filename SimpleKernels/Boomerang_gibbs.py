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
    x = x_ref
    v = np.dot(Sigma_sqrt, np.random.normal(0,1,dim))
    condition = np.random.normal(Mu_comp[2] + (Sigma_comp[2, 1] / Sigma_comp[1, 1]) * (x[1] - Mu_comp[1]),
                                 Sigma_comp[2, 2] - ((Sigma_comp[2, 1] * Sigma_comp[1, 2]) / Sigma_comp[1, 1]), 1)
    gradU = g_E(x, condition[0]) - np.dot(Sigma_inv, (x-x_ref)) # O(d^2) to compute
    gradBound = g_E(x_ref, condition[0]) # O(d^2) to compute
    M2 = np.sqrt(np.dot(gradBound, gradBound))
    updateSkeleton = False
    finished = False
    x_skeleton = x
    v_skeleton = v
    t_skeleton = np.array([t])


    dt_refresh = -np.log(np.random.rand())/refresh_rate


    rejected_switches = 0
    accepted_switches = 0
    phaseSpaceNorm = np.sqrt(np.dot(x-x_ref,x-x_ref) + np.dot(v,v))
    a = np.dot(v, gradU) #starting point for the rate
    b = M1 * phaseSpaceNorm**2 + M2 * phaseSpaceNorm #slope term for the rate

    errors = 0

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
        a = a + b*dt  #rate at the end of dt period -> simulated rate
        gradU = g_E(x, condition[0]) - np.dot(Sigma_inv, (x-x_ref)) # O(d^2) to compute, gradient at the end of dt period

        if not finished and dt_switch_proposed < dt_refresh:
            switch_rate = np.dot(v, gradU) # no need to take positive part, true rate at the end of dt period
            simulated_rate = a #bound at the end of dt period
            if simulated_rate < switch_rate:
                err = True
                print("simulated rate: ", simulated_rate)
                print("actual switching rate: ", switch_rate)
                #print("switching rate exceeds bound.")
                print("This is the intercept of the bounding line", a)
                print("This is the slope of the bounding line", b)
                print("This is the time at which exceeded", dt)
                print("This is the bound on the gradient", M2)
                print("This is the bound on the hessian", M1)
                print("This is the energy of the system", phaseSpaceNorm)
                print("This is the actual value", g_E(x_ref, condition[0]))
                errors = errors + 1
            if np.random.rand() * simulated_rate <= switch_rate:
                # obtain new velocity by reflection
                skewed_grad = np.dot(np.transpose(Sigma_sqrt), gradU)
                v = v - 2 * (switch_rate / np.dot(skewed_grad,skewed_grad)) * np.dot(Sigma_sqrt, skewed_grad)
                phaseSpaceNorm = np.sqrt(np.dot(x-x_ref,x-x_ref) + np.dot(v,v))
                #a = -switch_rate #origin for next step, note that after bounce only the sign of the inner product changes
                #print("a", a)
                #print("current switch rate", np.dot(v,gradU))
                #b = M1 * phaseSpaceNorm**2 + M2 * phaseSpaceNorm
                updateSkeleton = True
                accepted_switches += 1
            else:
                a = switch_rate # if not bounce -> the origin of the bounding line is still the same
                gradBound = g_E(x_ref, condition[0])  # O(d^2) to compute
                M2 = np.sqrt(np.dot(gradBound, gradBound))
                #a = np.dot(v, gradBound)
                b = M1 * phaseSpaceNorm ** 2 + M2 * phaseSpaceNorm
                updateSkeleton = False
                rejected_switches += 1

            # update refreshment time and switching time bound
            dt_refresh = dt_refresh - dt_switch_proposed

        if (not finished and dt_switch_proposed >= dt_refresh):
            # so we refresh
            updateSkeleton = True
            v = np.dot(Sigma_sqrt, np.random.normal(0,1,dim))
            phaseSpaceNorm = np.sqrt(np.dot(x-x_ref,x-x_ref) + np.dot(v,v))
            #a = np.dot(v, gradU)
            #b = M1 * phaseSpaceNorm**2 + M2 * phaseSpaceNorm

            # compute new refreshment time
            dt_refresh = -np.log(np.random.rand())/refresh_rate

        else:
            pass

        if updateSkeleton:
            x_skeleton = np.vstack((x_skeleton, np.transpose(x)))
            v_skeleton = np.vstack((v_skeleton, np.transpose(v)))
            t_skeleton = np.append(t_skeleton, t)
            condition = np.random.normal(Mu_comp[2] + (Sigma_comp[2,1] / Sigma_comp[1,1]) * (x[1] - Mu_comp[1]),Sigma_comp[2,2]-((Sigma_comp[2,1] * Sigma_comp[1,2])/Sigma_comp[1,1]),1)
            gradU = g_E(x, condition[0]) - np.dot(Sigma_inv, (x-x_ref))
            gradBound = g_E(x, condition[0])- np.dot(Sigma_inv, (x-x_ref))  # O(d^2) to compute
            M2 = np.sqrt(np.dot(gradBound, gradBound))
            a = np.dot(v, gradU)
            b = M1 * phaseSpaceNorm**2 + M2 * phaseSpaceNorm
            updateSkeleton = False

    return (t_skeleton, x_skeleton, v_skeleton, errors)


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

t, x, v, err = Boomerang_gibbs(boomerang_gradient, M1, 10000, x_ref, Sigma_inv_ref, refresh_rate=1.0)