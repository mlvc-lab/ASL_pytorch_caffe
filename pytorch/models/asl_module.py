import torch

import asl_cuda


class ASLFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, shift_param):
        output, = asl_cuda.forward(input, shift_param)
        variables = [input, shift_param]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, shift_param = ctx.saved_tensors
        outputs = asl_cuda.backward(input.contiguous(), grad_output.contiguous(), shift_param)
        grad_input, grad_weight = outputs

        return grad_input, grad_weight

class ActiveShiftLayer(torch.nn.Module):
    def __init__(self, in_channels):
        super(ActiveShiftLayer, self).__init__()
        self.in_channels = in_channels
        self.shift_param = torch.nn.Parameter(torch.Tensor(in_channels, 2))
        # Init
        self.shift_param.data.uniform_(-1, 1)

    def forward(self, input):
        return ASLFunction.apply(input, self.shift_param)

