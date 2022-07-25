import numpy as np
from boomerangJULIA import switchingtime, EllipticDynamics



def Boomerang(g_E, epochs, x_ref, Sigma_inv, refresh_rate=1.0, k=3):
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
    gradU = g_E(x) - np.dot(Sigma_inv, (x-x_ref)) # O(d^2) to compute

    updateSkeleton = False
    x_skeleton = x
    v_skeleton = v
    t_skeleton = np.array([t])

    dt_refresh = -np.log(np.random.rand())/refresh_rate

    rejected_switches = 0
    accepted_switches = 0
    bound_errors = 0
    slope = 0
    intercept = np.dot(v, gradU)
    clean = False
    rates = np.array([])
    bounds = np.array([])
    t_i = np.array([0])
    lambda_i = np.array([0])
    while len(t_skeleton)<epochs :
        if clean:
            t_i = np.array([0])
            lambda_i = np.array([0])
            slope = 0
            intercept = np.dot(v, gradU)
        dt_switch_proposed = switchingtime(slope, intercept*k)
        dt = np.minimum(dt_switch_proposed,dt_refresh)
        (y, v) = EllipticDynamics(dt, x-x_ref, v)
        x = y + x_ref
        t = t + dt
        gradU = g_E(x) - np.dot(Sigma_inv, (x-x_ref)) # O(d^2) to compute
        if  dt_switch_proposed < dt_refresh:
            switch_rate = np.dot(v, gradU) # no need to take positive part
            simulated_rate = slope * t + intercept + k
            rates = np.append(rates, switch_rate)
            bounds = np.append(bounds, simulated_rate)
            if simulated_rate < switch_rate:
                print("simulated rate: ", simulated_rate)
                print("actual switching rate: ", switch_rate)
                print("switching rate exceeds bound.")
                bound_errors = bound_errors + 1

            if np.random.rand() * simulated_rate <= switch_rate:
                # obtain new velocity by reflection
                skewed_grad = np.dot(np.transpose(Sigma_sqrt), gradU)
                v = v - 2 * (switch_rate / np.dot(skewed_grad,skewed_grad)) * np.dot(Sigma_sqrt, skewed_grad)
                updateSkeleton = True
                accepted_switches += 1
            else:
                    #prevent from zero adding
                t_i = np.append(t_i, dt_switch_proposed)
                lambda_i = np.append(lambda_i, switch_rate)
                #rejection -> update bound
                slope = (np.cov(lambda_i, t_i)[0,1])/np.var(t_i)
                intercept = np.mean(lambda_i) - slope * np.mean(lambda_i)
                updateSkeleton = False
                rejected_switches += 1

            # update refreshment time and switching time bound
            dt_refresh = dt_refresh - dt_switch_proposed

        if dt_switch_proposed >= dt_refresh:
            # so we refresh
            updateSkeleton = True
            v = np.dot(Sigma_sqrt, np.random.normal(0,1,dim))
            dt_refresh = -np.log(np.random.rand())/refresh_rate


        if updateSkeleton:
            clean = True
            x_skeleton = np.vstack((x_skeleton, np.transpose(x)))
            v_skeleton = np.vstack((v_skeleton, np.transpose(v)))
            t_skeleton = np.append(t_skeleton, t)
            updateSkeleton = False

    print("number of accepted switches: ", accepted_switches)
    print("number of proposed switches: ", accepted_switches + rejected_switches)
    #print(rejected_switches)
    return (t_skeleton, x_skeleton, v_skeleton, rates, bounds, bound_errors)

def boomerang_gradient(x):
    grad = np.dot(x-Mu, Sigma_inv)
    return grad

#TARGET VALUES
Sigma = np.array([[1,0],[0,1]])
Sigma_inv = np.linalg.inv(Sigma)
Mu = np.array([10,10])

"REFERENCE VALUES"
x_ref = np.zeros(2)
Sigma_inv_ref = Sigma_inv
M1 = np.linalg.norm(Sigma_inv)

t, x, v, rates, bounds, errors = Boomerang(boomerang_gradient, 10000, x_ref, Sigma_inv_ref, refresh_rate=1, k=2)