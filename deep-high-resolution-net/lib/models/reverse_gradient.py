from torch import nn
from torch.autograd import Function


class ReverseLayer(nn.Module):
    def __init__(self, stop_grad: bool = False):
        super().__init__()
        self.stop_grad = stop_grad

    def forward(self, x):
        return ReverseLayerF.apply(x, 1 - self.stop_grad)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        print(ctx.alpha)
        return grad_output.neg() * ctx.alpha, None