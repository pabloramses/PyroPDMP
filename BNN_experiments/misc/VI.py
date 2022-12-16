import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

x_train = -2*torch.pi*torch.rand(1000) + torch.pi
y_train = torch.sin(x_train) + torch.normal(0., 0.1, size=(1000,))

L = 2
class BNN(PyroModule):
  def __init__(self):
    super().__init__()

    # Hidden layer
    self.hidden = PyroModule[nn.Linear](1, L)
    # Weight priors
    #self.hidden.lambda_h_w = PyroSample(dist.HalfCauchy(1.).expand([L, 1]).to_event(2))
    #self.hidden.tau_h_w = PyroSample(dist.HalfCauchy(1))
    #self.hidden.weight = PyroSample(
      #lambda hidden: dist.Normal(0., (hidden.lambda_h_w) * (hidden.tau_h_w)).expand([L, 1]).to_event(2))
    self.hidden.weight = PyroSample(
      lambda hidden: dist.Normal(0., torch.ones((L,))).expand([L, 1]).to_event(2))
    # Bias priors
    #self.hidden.lambda_h_b = PyroSample(dist.HalfCauchy(1.).expand([L]).to_event(1))
    #self.hidden.tau_h_b = PyroSample(dist.HalfCauchy(1))
    #self.hidden.bias = PyroSample(
      #lambda hidden: dist.Normal(0., (hidden.lambda_h_b) * (hidden.tau_h_b)).expand([L]).to_event(1))
    self.hidden.bias = PyroSample(
      lambda hidden: dist.Normal(0., torch.ones((L,))).expand([L]).to_event(1))

    # Output layer
    self.out = PyroModule[nn.Linear](L, 1)
    # Weight priors
    #self.out.lambda_h_w = PyroSample(dist.HalfCauchy(1.).expand([1, L]).to_event(2))
    #self.out.tau_h_w = PyroSample(dist.HalfCauchy(1))
    #self.out.weight = PyroSample(
      #lambda out: dist.Normal(0., (out.lambda_h_w) * (out.tau_h_w)).expand([1, L]).to_event(2))
    self.out.weight = PyroSample(
      lambda out: dist.Normal(0., torch.ones((L,1))).expand([1, L]).to_event(2))
    # Bias priors
    #self.out.lambda_h_b = PyroSample(dist.HalfCauchy(1.).expand([1]).to_event(1))
    #self.out.tau_h_b = PyroSample(dist.HalfCauchy(1))
    #self.out.bias = PyroSample(lambda out: dist.Normal(0., (out.lambda_h_b) * (out.tau_h_b)).expand([1]).to_event(1))
    self.out.bias = PyroSample(
      lambda out: dist.Normal(0., torch.ones((1,))).expand([1]).to_event(1))

  def forward(self, x, y=None):
    x = F.relu(self.hidden(x))
    mu = self.out(x).squeeze()
    # Likelihood
    with pyro.plate("data", x.shape[0]):
      obs = pyro.sample("obs", dist.Normal(mu, 0.1), obs=y)
    return mu

bnn = BNN()
pyro.clear_param_store()

def guide(x,y):
    hidden_mean = pyro.param("hidden_weight", torch.zeros((L,)))
    out_mean = pyro.param("out_weight", torch.zeros((L,)))
    pyro.sample("hidden.weight", dist.Normal(hidden_mean, 1))
    pyro.sample("out.weight", dist.Normal(out_mean,1))


adam_params = {"lr":0.005, "betas":(0.90, 0.999)}
optimiser = Adam(adam_params)

svi = SVI(bnn, guide, optimiser, loss=Trace_ELBO())

for step in range(100):
  svi.step(x_train.reshape(-1,1), y_train.unsqueeze(1))
  if step % 10 == 0:
    print('.', end='')

#learned variational parametres
hidden_mean = pyro.param("hidden_weight").item()
out_mean = pyro.param("out_weight").item()

print(hidden_mean)
print(out_mean)