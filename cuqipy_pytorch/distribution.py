import torch
import cuqi
from cuqi.distribution import Distribution
import numbers
import numpy as np

class _OutOfBoundsDistribution:
    """ Helper class to handle out-of-bounds values """
    def log_prob(self, *args, **kwargs):
        return torch.tensor(-torch.inf, dtype=torch.float32)

class HalfGaussian(Distribution):
    def __init__(self, scale, is_symmetric=False, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 
        self.scale = scale

    @property
    def scale(self):
        return self._scale
        
    @scale.setter
    def scale(self, value):
        self._scale = value
        self._set_dist()
    
    def _set_dist(self):
        if hasattr(self, '_scale'):
            scale = self.scale
            # Set scale
            if isinstance(scale, numbers.Number):
                scale = scale*torch.ones(self.dim)
            self._dist = torch.distributions.HalfNormal(scale)

    def logpdf(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        return torch.sum(self._dist.log_prob(value))

    def _sample(self, n):
        return self._dist.sample(torch.Size((n,)))

class Uniform(Distribution):
    """ Uniform distribution """
    def __init__(self, low, high, is_symmetric=True, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 
        self.low = low
        self.high = high

    @property
    def low(self):
        return self._low
    
    @low.setter
    def low(self, low):
        self._low = low
        self._set_dist()

    @property
    def high(self):
        return self._high
    
    @high.setter
    def high(self, high):
        self._high = high
        self._set_dist()    

    def _set_dist(self):
        if hasattr(self, '_low') and hasattr(self, '_high'):
            low, high = self.low, self.high

            if isinstance(low, numbers.Number):
                low = low*torch.ones(self.dim)

            if isinstance(high, numbers.Number):
                high = high*torch.ones(self.dim)

            if isinstance(low, np.ndarray):
                low = torch.tensor(low)
            if isinstance(high, np.ndarray):
                high = torch.tensor(high)

            self._dist = torch.distributions.Uniform(low, high)

    def logpdf(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)

        if value < torch.tensor(self.low) or value > torch.tensor(self.high):
            return torch.tensor(-torch.inf, dtype=torch.float32)

        # Flip interval inclusion in logpdf, i.e. (low,high] instead of [low,high)
        if value == self.low: value = torch.tensor(self.high)
        elif value == self.high: value = torch.tensor(self.low)

        return torch.sum(self._dist.log_prob(value))

    def _sample(self, n):
        return self._dist.sample(torch.Size((n,)))

class LogGaussian(Distribution):
    def __init__(self, mean, cov, is_symmetric=False, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 
        self.mean = mean
        self.cov = cov

    @property
    def mean(self):
        return self._mean
    @mean.setter
    def mean(self, value):
        self._mean = value
        self._set_dist()

    @property
    def cov(self):
        return self._cov
    @cov.setter
    def cov(self, value):
        self._cov = value
        self._set_dist()
    
    def _set_dist(self):
        if hasattr(self, '_mean') and hasattr(self, '_cov'):
            mean = self.mean
            cov = self.cov
            # Set mean and cov to tensors if numbers
            if isinstance(mean, numbers.Number):
                mean = mean*torch.ones(self.dim)
            if isinstance(cov, numbers.Number):
                cov = cov*torch.ones(self.dim)

            if torch.is_tensor(mean) and torch.is_tensor(cov):
                self._dist = torch.distributions.LogNormal(mean, cov)

    def logpdf(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        #if isinstance(self._dist, Normal):
        return torch.sum(self._dist.log_prob(value))
        #else:
        #return self._dist.log_prob(value)

    def _sample(self, n):
        return self._dist.sample(torch.Size((n,)))

# Create a basic Gaussian distribution wrapping pytorch
class Gaussian(Distribution):
    def __init__(self, mean, cov, is_symmetric=True, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 
        self.mean = mean
        self.cov = cov

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean = value
        self._set_dist(update_only_mean=True)

    @property
    def cov(self):
        return self._cov
    
    @cov.setter
    def cov(self, value):
        self._cov = value
        self._set_dist()              

    def _set_dist(self, update_only_mean=False):
        """ Set the pytorch distribution if both values are specified """
        if hasattr(self, '_mean') and hasattr(self, '_cov'):
            # Define mean value
            if callable(self._mean) and hasattr(self._mean, '__len__'):
                mean = torch.zeros(len(self._mean))
            else:
                mean = self._mean

            # Define covariance value
            cov = self._cov

            # Set mean and cov to tensors if numbers
            if isinstance(mean, numbers.Number):
                mean = mean*torch.ones(self.dim)
            if isinstance(cov, numbers.Number):
                cov = cov*torch.ones(self.dim)

            if isinstance(mean, np.ndarray):
                mean = torch.tensor(mean)
            if isinstance(cov, np.ndarray):
                cov = torch.tensor(cov)

            # If both are tensors we create dist
            if torch.is_tensor(mean) and torch.is_tensor(cov):

                #if torch.isnan(mean).any():
                #    raise ValueError("mean contains NaN")
                #if torch.isnan(cov).any():
                #    raise ValueError("cov contains NaN")
                    
                # Special update for mean value to speed-up computation
                if hasattr(self, '_dist') and update_only_mean:
                    if cov.ndim==1:
                        self._dist.loc = self._mean.expand(self._dist.batch_shape)
                    else:
                        self._dist.loc = self._mean.expand(self._dist.batch_shape + (-1,)) 
                elif cov.ndim==1: #Make i.i.d. Gaussian
                    sqrt_cov = torch.sqrt(cov)
                    if torch.isnan(sqrt_cov).any():
                        self._dist = _OutOfBoundsDistribution()
                    else:
                        self._dist = torch.distributions.Normal(mean, sqrt_cov)
                else:
                    self._dist = torch.distributions.MultivariateNormal(mean, cov)
                    
    def logpdf(self, value):
        if isinstance(value, np.ndarray):
            value = torch.tensor(value)
        if isinstance(self._dist, torch.distributions.Normal):
            return torch.sum(self._dist.log_prob(value))
        else:
            return self._dist.log_prob(value)

    def _sample(self, n):
        return self._dist.sample(torch.Size((n,)))

    def gradient(self, v1, v2=None):
        if v2 is None:              #Prior case
            v1.requires_grad = True
            v1.grad = None
            Q = self.logpdf(v1)     # Forward pass
            Q.backward()            # Backward pass
            return v1.grad
        else:                       #Likelihood case
            v2.requires_grad = True
            v2.grad = None
            Q = self(v2).logpdf(v1) # Forward pass
            Q.backward()            # Backward pass
            return v2.grad

# Create a basic Gaussian distribution wrapping pytorch
class Gamma(Distribution):
    def __init__(self, shape, rate, is_symmetric=False, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 
        self.shape = shape
        self.rate = rate

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor([value], dtype=torch.float32)
        self._shape = value
        self._set_dist()

    @property
    def rate(self):
        return self._rate
    
    @rate.setter
    def rate(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor([value], dtype=torch.float32)
        self._rate = value
        self._set_dist()              

    def _set_dist(self):
        """ Set the pytorch distribution if both values are specified """
        if hasattr(self, '_shape') and hasattr(self, '_rate'):
            # Define shape value
            shape = self._shape

            # Define rate value
            rate = self._rate

            # If both are tensors we create dist
            if torch.is_tensor(shape) and torch.is_tensor(rate):
                self._dist = torch.distributions.Gamma(shape, rate)
                    
    def logpdf(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor([value], dtype=torch.float32)
        # Check if value is negative
        if value.min() <= 0:
            return -float('inf')
        return torch.sum(self._dist.log_prob(value))

    def _sample(self, n):
        return self._dist.sample(torch.Size((n,)))

class StackedJointDistribution(cuqi.distribution._StackedJointDistribution, Distribution):
    
    def logpdf(self, x):
        # Cache x
        self._x_np = x

        # Convert x to tensor
        self._x = torch.tensor(x, requires_grad=True)

        # Evaluate logpdf (and cache)
        self._logpdf = self.logd(self._x)

        # Return as numpy
        return self._logpdf.detach().numpy()

    def gradient(self, x):
        if hasattr(self, '_x_np') and self._x_np is x:
            self._logpdf.backward()
            return self._x.grad.detach().numpy()
        else:
            self.logpdf(x)
            return self.gradient(x)

    def _sample(self, Ns):
        pass


# Create a basic Gaussian distribution wrapping pytorch
class Gaussian2(Distribution):
    def __init__(self, mean, sqrtcov, is_symmetric=True, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 
        self.mean = mean
        self.sqrtcov = sqrtcov

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean = value
        self._set_dist(update_only_mean=True)

    @property
    def sqrtcov(self):
        return self._sqrtcov
    
    @sqrtcov.setter
    def sqrtcov(self, value):
        self._sqrtcov = value
        self._set_dist()              

    def _set_dist(self, update_only_mean=False):
        """ Set the pytorch distribution if both values are specified """
        if hasattr(self, '_mean') and hasattr(self, '_sqrtcov'):
            # Define mean value
            if callable(self._mean) and hasattr(self._mean, '__len__'):
                mean = torch.zeros(len(self._mean))
            else:
                mean = self._mean

            # Define covariance value
            sqrtcov = self._sqrtcov

            # If both are tensors we create dist
            if torch.is_tensor(mean) and torch.is_tensor(sqrtcov):
                    
                # Special update for mean value to speed-up computation
                if hasattr(self, '_dist') and update_only_mean:
                    if sqrtcov.ndim==1:
                        self._dist.loc = self._mean.expand(self._dist.batch_shape)
                    else:
                        self._dist.loc = self._mean.expand(self._dist.batch_shape + (-1,)) 
                elif sqrtcov.ndim==1: #Make i.i.d. Gaussian
                    if torch.isnan(sqrtcov).any():
                        self._dist = _OutOfBoundsDistribution()
                    else:
                        self._dist = torch.distributions.Normal(mean, sqrtcov)
                else:
                    self._dist = torch.distributions.MultivariateNormal(mean, scale_tril=sqrtcov)
                    
    def logpdf(self, value):
        if isinstance(self._dist, torch.distributions.Normal):
            return torch.sum(self._dist.log_prob(value))
        else:
            return self._dist.log_prob(value)

    def _sample(self, n):
        return self._dist.sample(torch.Size((n,)))

    def gradient(self, v1, v2=None):
        if v2 is None:              #Prior case
            v1.requires_grad = True
            v1.grad = None
            Q = self.logpdf(v1)     # Forward pass
            Q.backward()            # Backward pass
            return v1.grad
        else:                       #Likelihood case
            v2.requires_grad = True
            v2.grad = None
            Q = self(v2).logpdf(v1) # Forward pass
            Q.backward()            # Backward pass
            return v2.grad
