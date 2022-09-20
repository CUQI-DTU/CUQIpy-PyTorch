from cuqi.model import Model
from torch.autograd import Function
from torch.autograd.function import once_differentiable

def add_to_autograd(model: Model):
    """ Add given forward model to torch autograd.
    
    Currently forgets about its geometry. This will be fixed in a future version.
    """

    # Create custom auto-grad function from model
    class CustomFunction(Function):
        @staticmethod
        def forward(ctx, input):
            numpy_input = input.detach().numpy()
            output = model.forward(numpy_input)
            ctx.save_for_backward(input)
            return input.new(output)

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            numpy_grad_output = grad_output.detach().numpy()
            numpy_input = input.detach().numpy()
            grad_input = model.gradient(numpy_grad_output, numpy_input)
            return input.new(grad_input)

    torch_forward = CustomFunction.apply

    # Add to cuqi model
    torch_model = Model(torch_forward, int(model.range_dim), int(model.domain_dim))

    return torch_model
