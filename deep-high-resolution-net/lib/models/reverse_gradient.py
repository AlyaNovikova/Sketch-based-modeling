from torch import nn
from torch.autograd import Function


class ReverseLayer(nn.Module):
    def forward(self, x):
        return ReverseLayerF.apply(x)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()
