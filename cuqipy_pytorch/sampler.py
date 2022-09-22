import pyro
from cuqi.samples import Samples
from cuqi.distribution import Posterior, JointDistribution
from cuqipy_pytorch.distribution import StackedJointDistribution
import torch as xp
import numpy as np

class NUTS:

    def __init__(self, target, MCMC_kwargs=None, NUTS_kwargs=None):

        if MCMC_kwargs is None:
            MCMC_kwargs = {}
        if NUTS_kwargs is None:
            NUTS_kwargs = {}

        if isinstance(target, StackedJointDistribution):
            self.joint = JointDistribution(*target._densities)
            self.target = target
        elif isinstance(target, JointDistribution):
            self.joint = target
            self.target = StackedJointDistribution(*target._densities)
        elif isinstance(target, Posterior):
            self.joint = JointDistribution(target.likelihood, target.prior)
            self.target = StackedJointDistribution(*self.joint._densities)

        self.kernel = pyro.infer.NUTS(potential_fn=self.potential_fn, **NUTS_kwargs)

        self.MCMC_kwargs = MCMC_kwargs

    def potential_fn(self, params):
        return -self.target.logd(params["x"])

    def sample(self, N, Nb=200):

        # NUTS kernel
        kernel = self.kernel

        # Add defaults to MCMC kwargs
        MCMC_kwargs = self.MCMC_kwargs
        MCMC_kwargs["kernel"] = kernel
        MCMC_kwargs["num_samples"] = N
        MCMC_kwargs["warmup_steps"] = Nb
        MCMC_kwargs["num_chains"] = 1
        if "initial_params" not in MCMC_kwargs:
            MCMC_kwargs["initial_params"] = {"x":xp.ones(self.target.dim)}

        # Sampler
        sampler = pyro.infer.MCMC(**MCMC_kwargs)
        
        # Run sampler
        sampler.run()

        # Get samples
        samples_pyro = sampler.get_samples()

        # Split samples
        split_indices = np.cumsum(self.joint.dim)
        split_samples = np.split(samples_pyro["x"].numpy().T, split_indices[:-1])

        # Acc to dict
        samples = {}
        par_names = self.joint.get_parameter_names()
        for i, par_name in enumerate(par_names):
            samples[par_name] = Samples(split_samples[i], self.target.get_density(par_name).geometry)

        #if len(samples) == 1:
        #    return samples[par_names[0]]
        return samples
