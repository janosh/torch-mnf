import torch
from torch import nn

from torch_mnf.models import MLP


class RNVP(nn.Module):
    """Affine half flow aka Real Non-Volume Preserving (x = z * exp(s) + t),
    where a randomly selected half z1 of the dimensions in z are transformed as an
    affine function of the other half z2, i.e. scaled by s(z2) and shifted by t(z2).

    From "Density estimation using Real NVP", Dinh et al. (May 2016)
    https://arxiv.org/abs/1605.08803

    This implementation uses the numerically stable updates introduced by IAF:
    https://arxiv.org/abs/1606.04934
    """

    def __init__(self, dim, h_sizes=(30,)):
        super().__init__()
        self.net = MLP(dim, *h_sizes)
        self.t = nn.Linear(h_sizes[-1], dim)
        self.s = nn.Linear(h_sizes[-1], dim)

    def forward(self, z):  # z -> x
        # Get random Bernoulli mask. This decides which channels will remain
        # unchanged and which will be transformed as functions of the unchanged.
        mask = torch.bernoulli(0.5 * torch.ones_like(z))

        z1, z2 = (1 - mask) * z, mask * z
        y = self.net(z2)
        shift, scale = self.t(y), self.s(y)

        # sigmoid(x) = 1 / (1 + exp(-x)). For x in (-inf, inf) => sigmoid(x) in (0, 1).
        gate = torch.sigmoid(scale)
        log_dets = ((1 - mask) * gate.log()).sum(1)
        x = (z1 * gate + (1 - gate) * shift) + z2

        return x, log_dets
