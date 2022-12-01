"""
NUTS sampling with non-Gaussian likelihood
==========================================
Demonstrates how to use NUTS sampling with a non-Gaussian likelihoods.

Also demonstrates how to add CUQIpy models to the PyTorch autograd engine.
"""

# %% Imports
import torch as xp
import cuqi

from cuqipy_pytorch.distribution import Cauchy, Laplace, Gaussian
from cuqipy_pytorch.model import add_to_autograd
from cuqipy_pytorch.sampler import NUTS

# %% Forward models
n = 50

# Testproblem
TP = cuqi.testproblem.Deconvolution1D(dim=n)

# CUQIpy model and data added to autograd framework. y = Ax
A = add_to_autograd(TP.model)
y_obs = xp.as_tensor(TP.data)

# Lambda function as forward model. d = Bu
B = lambda u: xp.log(u**2 + 1) + xp.sin(u) # Contrived example

# %% Bayesian models (non-Gaussian likelihoods)

# Model 1
x = Gaussian(xp.zeros(n), 0.1)                     # x ~ N(0, 1)
y = Cauchy(location=A(x), scale=0.1)               # y ~ Cauchy(Ax, 0.1)
J1 = cuqi.distribution.JointDistribution(x, y)

# Model 2
u = Gaussian(xp.zeros(n), 1)                    # u ~ N(0, 1)
d = Laplace(location=B, scale=0.1*xp.ones(n))   # d ~ Laplace(Bu, 0.1)
J2 = cuqi.distribution.JointDistribution(u, d)

# %% Posteriors
print("Posterior 1:")
P1 = J1(y=y_obs)
print(P1)

print("Posterior 2:")
P2 = J2(d=xp.ones(n))
print(P2)

# %% Evaluating logpdf's
print(P1.logpdf(x=xp.ones(n)))
print(P2.logpdf(u=xp.ones(n)))

# %% Sampling model 2 with NUTS

samples2 = NUTS(P2).sample(100, 100)

# %% Sampling model 1 with NUTS

samples1 = NUTS(P1).sample(100, 100) 

# %% Compare "Data" (ones) with posterior mean through forward model
# Value should be close to 1
print(B(xp.as_tensor(samples2["u"].mean())))

# %% Posterior plot for model 1
samples1["x"].plot_ci(exact=TP.exactSolution)

# %% Sampling without explicit creation of joint and posterior distributions
# This is a convenience function that does the same as above.
# Namely, it creates a joint distribution and posterior distribution
# and then samples from the posterior distribution using NUTS.

def sample_posterior(*densities, n_samples=500, n_burnin=500, **data):
    """ Sample posterior given by a list of densities. The observations are given as keyword arguments. """
    P = cuqi.distribution.JointDistribution(*densities)
    return NUTS(P(**data)).sample(n_samples, n_burnin)

# %% Example with convenience function

samples3 = sample_posterior(x, y, y=y_obs, n_samples=100, n_burnin=100)

# %% Posterior plot for model 1 with samples from convenience function
samples3["x"].plot_ci(exact=TP.exactSolution)
# %%
