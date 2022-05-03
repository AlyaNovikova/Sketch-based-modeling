from torch import nn
from torch.autograd import Function


class ReverseLayer(nn.Module):
    def forward(self, x, stop_grad: bool = False):
        return ReverseLayerF.apply(x, 1 - stop_grad)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None