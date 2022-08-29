import torch
from pyro.ops.integrator import potential_grad
from progress.bar import Bar

def predictive_samples(posterior_samples, points_to_evaluate):
  predictive_sample = torch.matmul(posterior_samples[0,:], points_to_evaluate[0,:])
  for j in range(1,posterior_samples.shape[0]):
    predictive_sample = torch.hstack((predictive_sample, torch.matmul(posterior_samples[j,:], points_to_evaluate[0,:])))

  with Bar('Predictive sampling ', max=points_to_evaluate.shape[0]-1) as bar:
    for i in range(1, points_to_evaluate.shape[0]):
      predictive_observation = torch.matmul(posterior_samples[0,:], points_to_evaluate[i,:])
      for j in range(1, posterior_samples.shape[0]):
        predictive_observation = torch.hstack((predictive_observation, torch.matmul(posterior_samples[j,:], points_to_evaluate[i,:])))
      predictive_sample = torch.vstack((predictive_sample, predictive_observation))
      bar.next()
  return predictive_sample

def predictive_summary(predictive_sample, alpha):
  m = predictive_sample.shape[1]
  lk = int(m*alpha)
  uk = int(m*(1-alpha))
  mean = torch.mean(predictive_sample[0])
  lower = torch.sort(predictive_sample[0])[0][lk]
  upper = torch.sort(predictive_sample[0])[0][uk]
  for i in range(1,predictive_sample.shape[0]):
    mean = torch.hstack((lower, torch.mean(predictive_sample[i])))
    lower = torch.hstack((lower, torch.sort(predictive_sample[i])[0][lk]))
    upper = torch.hstack((upper, torch.sort(predictive_sample[i])[0][uk]))

  return mean,lower, upper

def percentage_correct(lower, upper, true):
  count = 0
  total = lower.shape[0]
  for i in range(total):
    if lower[i]<=true[i]<=upper[i]:
      count +=1
  return count/total



def l2_norm(x):
  return x.pow(2).sum(-1).sqrt()

beta = -0.5
c = 1000

def kernel(theta, thetap):
  return (c ** 2 + l2_norm(theta - thetap)) ** (beta)

def kernel_gradient_j(theta, theta_prime, j):
  return beta * (kernel(theta,theta_prime))**((beta-1)/beta) * (torch.sum((theta-theta_prime)**2)**(-0.5)) * (theta[j]-theta_prime[j])

def kernel_hessian_jj(theta, theta_prime, j):
  ss = torch.sum((theta-theta_prime)**2)
  return -beta * (beta-1) * (kernel(theta, theta_prime)**((beta-2)/beta)) * (ss**(-1)) * ((theta[j]-theta_prime[j])**2) + beta * (kernel(theta,theta_prime))**((beta-1)/beta) * ((ss**(-3/2)) * ((theta[j]-theta_prime[j])**2) - (ss**(-0.5)))

def kernel_Stein_Discrepancies(mcmc_kernel, sample):
  # Computation of ALL the grads
  dim = sample.shape[1]
  z_i = {'beta': sample[0]}

  z_grad, en = potential_grad(mcmc_kernel.potential_fn, z_i)
  grads = z_grad['beta']
  for i in range(1, sample.shape[0]):
    z_i = {'beta': sample[i]}
    z_grad, en = potential_grad(mcmc_kernel.potential_fn, z_i)
    grads = torch.vstack((grads, z_grad['beta']))

  K = sample.shape[0]
  KSD = torch.tensor(0)

  for d in range(dim):
    KSD_d = 0
    with Bar('Dimension ' + str(d), max = K) as bar:
      for i in range(K):
        theta = sample[i]
        grad_u_d = grads[i][d]
        for j in range(i, K):
          theta_prime = sample[j]
          grad_u_d_prime = grads[j][d]
          grad_kernel_d = kernel_gradient_j(theta, theta_prime, d)
          grad_kernel_prime_d = -grad_kernel_d
          hess_d_dprime = kernel_hessian_jj(theta, theta_prime, d)
          new = ((grad_u_d * grad_u_d_prime) * kernel(theta,theta_prime) + grad_u_d * grad_kernel_prime_d + grad_u_d_prime * grad_kernel_d + hess_d_dprime)/(K**2)
          m = 1 + 1 * (i != j)
          if torch.isnan(new):
            KSD_d = KSD_d
          else:
            KSD_d = KSD_d + m*new
        bar.next()
    KSD = KSD + torch.sqrt(KSD_d)


  return torch.sqrt(torch.abs(KSD))

def get_i_values(dictionary, i):
  keys = list(dictionary.keys())
  z_i = {keys[0]: dictionary[keys[0]][i]}
  t_i = dictionary[keys[0]][i]
  for k in range(1,len(keys)):
    new_i = dictionary[keys[k]][i]
    z_i.update({keys[k]:new_i})
    if len(new_i.shape)==0:
      t_i = torch.cat((t_i, new_i.reshape(1)))
    else:
      t_i = torch.cat((t_i, new_i))
  return z_i, t_i

def concatenate_keys(dictionary):
  keys = list(dictionary.keys())
  t_i = dictionary[keys[0]]
  for k in range(1,len(keys)):
    new_i = dictionary[keys[k]]
    if len(new_i.shape)==0:
      t_i = torch.cat((t_i, new_i.reshape(1)))
    else:
      t_i = torch.cat((t_i, new_i))
  return t_i

def KSD_hp(mcmc_kernel, sample, K=1000, c=100, beta=0.5):
  # Computation of ALL the grads
  #dim = mcmc_kernel.dim
  c2 = c ** 2
  keys = list(sample.keys())
  n = sample[keys[0]].shape[1]
  imq_ksd_sum = 0

  # Calculate KSD
  with Bar('Convergence ', max=K) as bar:
    for i in range(n-K, n):
      z1, x1 = get_i_values(sample, i)
      z1_grad, en = potential_grad(mcmc_kernel.potential_fn, z1)
      gradlogp1 = concatenate_keys(z1_grad)
      if i==n-K:
        dim = len(x1)
      for j in range(i, n):
        z2, x2 = get_i_values(sample, j)
        z2_grad, en = potential_grad(mcmc_kernel.potential_fn, z2)
        gradlogp2 = concatenate_keys(z2_grad)
        diff = x1 - x2
        diff2 = torch.sum(diff ** 2)

        base = diff2 + c2
        base_beta = base ** (-beta)
        base_beta1 = base_beta / base

        kterm_sum = torch.sum(gradlogp1 * gradlogp2) * base_beta
        coeffgrad = -2.0 * beta * base_beta1
        gradx1term_sum = torch.sum(gradlogp1 * (-diff)) * coeffgrad
        gradx2term_sum = torch.sum(gradlogp2 * diff) * coeffgrad
        gradx1x2term_sum = (-dim + 2 * (beta + 1) * diff2 / base) * coeffgrad
        m = 1 + 1 * (i != j)
        imq_ksd_sum = imq_ksd_sum + m * (kterm_sum + gradx1term_sum + gradx2term_sum + gradx1x2term_sum)
      bar.next()
  imq_ksd = torch.sqrt(imq_ksd_sum) / K
  return (imq_ksd)


def sparsity(sparse_coefs, posterior_samples, threshold=0.2):
  pruned = []
  n = posterior_samples.shape[0]
  for i in range(n):
    if posterior_samples[i] < threshold:
      pruned.append(0)
    else:
      pruned.append(1)
  return torch.mean((torch.abs(torch.Tensor(pruned)) == sparse_coefs)*1.0)


def KSD_hp_2(mcmc_kernel, sample, K=1000, c=1000, beta=0.7):
  # Computation of ALL the grads
  #dim = mcmc_kernel.dim
  c2 = c ** 2
  keys = list(sample.keys())
  n = sample[keys[0]].shape[1]
  imq_ksd_sum = 0
  z_i, x = get_i_values(sample, 0)
  zi_grad, en = potential_grad(mcmc_kernel.potential_fn, z_i)
  grad = concatenate_keys(zi_grad)

  with Bar('Retrieving gradients', max=K) as bar:
    for i in range(n-K, n):
      z_i, x_i = get_i_values(sample, i)
      x = torch.vstack((x,x_i))
      zi_grad, en = potential_grad(mcmc_kernel.potential_fn, z_i)
      grad_i = concatenate_keys(zi_grad)
      grad = torch.vstack((grad, grad_i))
      bar.next()

  # Calculate KSD
  with Bar('Computing KSD ', max=K) as bar:
    for i in range(n-K, n):
      x1 = x[i,]
      gradlogp1 = grad[i,]
      if i==n-K:
        dim = len(x1)
      for j in range(i, n):
        x2 = x[j,]
        gradlogp2 = grad[j,]
        diff = x1 - x2
        diff2 = torch.sum(diff ** 2)

        base = diff2 + c2
        base_beta = base ** (-beta)
        base_beta1 = base_beta / base

        kterm_sum = torch.sum(gradlogp1 * gradlogp2) * base_beta
        coeffgrad = -2.0 * beta * base_beta1
        gradx1term_sum = torch.sum(gradlogp1 * (-diff)) * coeffgrad
        gradx2term_sum = torch.sum(gradlogp2 * diff) * coeffgrad
        gradx1x2term_sum = (-dim + 2 * (beta + 1) * diff2 / base) * coeffgrad
        m = 1 + 1 * (i != j)
        new = (kterm_sum + gradx1term_sum + gradx2term_sum + gradx1x2term_sum)
        if torch.isnan(new):
          pass
        else:
          imq_ksd_sum = imq_ksd_sum + m * (kterm_sum + gradx1term_sum + gradx2term_sum + gradx1x2term_sum)
      bar.next()
  imq_ksd = torch.sqrt(imq_ksd_sum) / K
  return (imq_ksd)