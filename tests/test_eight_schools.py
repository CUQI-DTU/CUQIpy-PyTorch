import torch as xp
from cuqi.distribution import JointDistribution
from cuqipy_pytorch.distribution import Gaussian, Lognormal
from cuqipy_pytorch.sampler import NUTS

def test_eight_schools_NUTS():
    """ This tests the pyro NUTS sampler on the eight schools example. """

    # Observations
    y_obs = xp.tensor([28, 8, -3,  7, -1, 1,  18, 12], dtype=xp.float32)
    σ_obs = xp.tensor([15, 10, 16, 11, 9, 11, 10, 18], dtype=xp.float32)

    # Bayesian model
    μ     = Gaussian(0, 10**2)
    τ     = Lognormal(5, 1)
    θp    = Gaussian(xp.zeros(8), 1)
    θ     = lambda μ, τ, θp: μ+τ*θp
    y     = Gaussian(θ, cov=σ_obs**2)

    # Posterior sampling
    joint = JointDistribution(μ, τ, θp, y)   # Define joint distribution 
    posterior = joint(y=y_obs)               # Define posterior distribution
    sampler = NUTS(posterior)                # Define sampling strategy
    samples = sampler.sample(N=100, Nb=100)  # Sample from posterior

    # Check shape of samples fits
    assert samples["μ"].shape == (1, 100)
    assert samples["τ"].shape == (1, 100)
    assert samples["θp"].shape == (8, 100)
