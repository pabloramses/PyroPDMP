# PyroPDMP

PyroPDMP is an in development repository built on [Pyro](https://github.com/pyro-ppl/pyro) that implements the simulation of [Piece-wise Deterministic Markov Processes](https://www.jstor.org/stable/26771007) as a continuous-time Monte Carlo-Markov Chain method for Bayesian Inference, using the flexibility to define models without the need to specify gradients that Pyro provides.

## Content

PyroPDMP currently includes the following PDMP samplers: 

- [Bouncy Particle Sampler (BPS)](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1294075)
- [Zig-Zag Sampler (ZZ)](https://arxiv.org/abs/1607.03188)
- [Boomerang Sampler](https://proceedings.mlr.press/v119/bierkens20a/bierkens20a.pdf)

The above can be implemented 
