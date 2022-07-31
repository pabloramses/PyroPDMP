import numpy as np
from boomerangJULIA import Boomerang

"CREATING THE DATA"
x0 = np.ones(100)
x1 = np.linspace(0,1,100)
X_design = np.vstack((x0,x1)).transpose()
true_beta = np.array([2,3])
sigma = 0.01
y = np.dot(X_design, true_beta) + np.random.normal(0, sigma, 100)

sigma_prior_est = 0.2

"TUNING THE REFERENCE MEASURE = PRIOR"
x_ref = np.zeros(2)
Sigma_ref_inv = np.eye(2)

"POSTERIOR TARGET"

Target_sigma = np.linalg.inv(Sigma_ref_inv + (1/sigma_prior_est) * np.dot(X_design.transpose(), X_design))

Target_sigma_inv = np.linalg.inv(Target_sigma)

Target_mu = np.dot(np.linalg.inv((Sigma_ref_inv + (1/sigma_prior_est) * np.dot(X_design.transpose(), X_design))),(1/sigma_prior_est)* (np.dot(y, X_design)))

M1 = np.linalg.norm(Target_sigma_inv)

"GRADIENT OF E(x)"

def reg_gradient(x):
    grad = np.dot((x-Target_mu), Target_sigma_inv)
    return grad

"SAMPLING"

(t_skeleton, x_skeleton, v_skeleton, x_ref) = Boomerang(reg_gradient, M1, 1000.0, x_ref, Sigma_ref_inv, refresh_rate=1.0)