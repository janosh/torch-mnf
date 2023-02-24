import torch
from torch import nn


class AffineConstantFlow(nn.Module):
    """Scales + Shifts the flow by (learned) constants per dimension. The only reason
    to have this layer is that the NICE paper defines a scaling-only layer which
    is a special case of this where t is zero (shift=False).
    """

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim)) if scale else torch.zeros(1, dim)
        self.t = nn.Parameter(torch.randn(1, dim)) if shift else torch.zeros(1, dim)

    def forward(self, z):
        x = z * torch.exp(self.s) + self.t
        log_det = torch.sum(self.s, dim=1)
        return x, log_det

    def inverse(self, x):
        z = (x - self.t) * torch.exp(-self.s)
        log_det = torch.sum(-self.s, dim=1)
        return z, log_det


class ActNormFlow(AffineConstantFlow):
    """Really an AffineConstantFlow but with activation normalizaton (similar
    to batch normalization), a data-dependent initialization, where on
    the very first batch we cleverly initialize the scale and translate
    function (s, t) so that the output is unit Gaussian. After initialization,
    the scale and bias are treated as regular trainable params that are data
    independent. See Glow paper sec. 3.1. https://arxiv.org/abs/1807.03039.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False

    def inverse(self, x):
        # first batch is used for init
        if self.data_dep_init_done is False:
            if not all(self.s.squeeze() == 0):
                self.s.data = x.std(dim=0, keepdim=True).log().detach()
            if not all(self.t.squeeze() == 0):
                self.t.data = (x * self.s.exp()).mean(dim=0, keepdim=True).detach()
            self.data_dep_init_done = True
        return super().inverse(x)
