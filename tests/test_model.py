import cuqi
from cuqipy_pytorch.model import add_to_autograd
import torch as xp
import numpy as np

def test_add_to_autograd():

    model_cuqi = cuqi.testproblem.Deconvolution1D().model

    # Add model to autograd framework
    model_torch = add_to_autograd(model_cuqi)

    # Check that the model gives the same result as the original
    x = np.random.randn(model_cuqi.domain_dim)

    assert np.allclose(model_cuqi(x), model_torch(xp.as_tensor(x)))

    # and that parameter name matches

    assert model_cuqi._non_default_args == model_torch._non_default_args
