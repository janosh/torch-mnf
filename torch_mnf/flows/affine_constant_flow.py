import torch
from torch import nn


class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a scaling layer which is a special case
    of this where t is None.
    """

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.scale, self.shift = scale, shift
        self.s = nn.Parameter(torch.randn(1, dim)) if scale else torch.zeros(1, dim)
        self.t = nn.Parameter(torch.randn(1, dim)) if scale else torch.zeros(1, dim)

    def forward(self, x):
        z = x * torch.exp(self.s) + self.t
        log_det = torch.sum(self.s, dim=1)
        return z, log_det

    def inverse(self, z):
        x = (z - self.t) * torch.exp(-self.s)
        log_det = torch.sum(-self.s, dim=1)
        return x, log_det
