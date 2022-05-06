from torch import nn
from torch.autograd import Function


class ReverseLayer(nn.Module):
    def __init__(self, stop_grad: bool = False):
        super().__init__()
        self.stop_grad = stop_grad

    def forward(self, x):
        my_print('!!!!!!!')
        return ReverseLayerF.apply(x, 1 - self.stop_grad)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        my_print('forward ', alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        my_print('backward', ctx.alpha)
        return grad_output.neg() * ctx.alpha, None


def my_print(*args):
    with open('super_output', 'wa') as f:
        f.write(' '.join(str(arg) for arg in args) + '\n')