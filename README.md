# PyroPDMP

PyroPDMP is an in development repository built on [Pyro](https://github.com/pyro-ppl/pyro) that implements the simulation of [Piece-wise Deterministic Markov Processes](https://www.jstor.org/stable/26771007) as a continuous-time Monte Carlo-Markov Chain method for Bayesian Inference, using the flexibility to define models without the need to specify gradients that Pyro provides.

## Content

PyroPDMP currently includes the following PDMP samplers: 

- [Bouncy Particle Sampler (BPS)](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1294075)
- [Zig-Zag Sampler (ZZ)](https://arxiv.org/abs/1607.03188)
- [Boomerang Sampler](https://proceedings.mlr.press/v119/bierkens20a/bierkens20a.pdf)

For the simulation of PDMPs it is necessary to sample from an In-Homogeneous Poisson Process. This can either be done by integrating the jump rates of the PDMP or by bounding them. Several approaches are presently available to simulate event times: 

- Dominated gradient/hessian based strategies. 
- [Numerical Integration of the rates](https://arxiv.org/abs/2003.03636)
- [Numerical optimisation to bound the rates](https://arxiv.org/abs/2206.11410) 

Several performance and diagnosis metrics are available as well (any of the available in Pyro can indeed be used). Of particular interest, it includes a simple implementation of a [Kernel Stein Discrepancy](https://arxiv.org/pdf/1907.06986.pdf) measure that is very well suited for gradient-based MCMC methods.  
