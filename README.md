# CUQIpy-PyTorch

CUQIpy-PyTorch is a plugin for the [CUQIpy](https://github.com/CUQI-DTU/CUQIpy) software package.

It adds a PyTorch backend to CUQIpy, allowing the user to use the PyTorch API to define models, distributions etc.

It also links to the [Pyro](https://pyro.ai/) No U-Turn Hamiltonian Monte Carlo sampler (NUTS) for efficient sampling from the joint posterior.

## Installation
For optimal performance consider installing [PyTorch](https://pytorch.org/) using conda, then install CUQIpy-PyTorch using pip:
```bash
pip install cuqipy-pytorch
```
If PyTorch, Pyro or CUQIpy are not installed, they will be installed automatically from the above command.

## Quickstart
Example for sampling from the [eight schools model](https://github.com/blei-lab/edward/blob/master/notebooks/eight_schools.ipynb):

$$
\begin{align*}
    \mu &\sim \mathcal{N}(0, 10^2)\\
    \tau &\sim \log\mathcal{N}(5, 1)\\
    \boldsymbol \theta' &\sim \mathcal{N}(\mathbf{0}, \mathbf{I}_m)\\
    \boldsymbol \theta &= \mu + \tau \boldsymbol \theta'\\
    \mathbf{y} &\sim \mathcal{N}(\boldsymbol \theta, \boldsymbol \sigma^2 \mathbf{I}_m)
\end{align*}
$$

where $\mathbf{y}\in\mathbb{R}^m$ and $\boldsymbol \sigma\in\mathbb{R}^m$ are observed data.

```python
import torch as xp
from cuqi.distribution import JointDistribution
from cuqipy_pytorch.distribution import Gaussian, Lognormal
from cuqipy_pytorch.sampler import NUTS

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
samples = sampler.sample(N=500, Nb=500)  # Sample from posterior

# Plot posterior samples
samples["θp"].plot_violin(); 
print(samples["μ"].mean()) # Average effect
print(samples["τ"].mean()) # Average variance
```

For more examples, see the [demos](demos) folder.

