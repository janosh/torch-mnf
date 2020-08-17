"""
NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516

Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770

Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)
"""

import torch
from torch import nn

from torch_mnf.models import MLP


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True):
        super().__init__()
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), dim // 2)
        if scale:
            self.s_cond = net_class(dim // 2, dim // 2, nh)
        if shift:
            self.t_cond = net_class(dim // 2, dim // 2, nh)

    def forward(self, x, inverse=False):
        x0, x1 = x.chunk(2, dim=1)
        if self.parity:
            x0, x1 = x1, x0
        s, t = self.s_cond(x0), self.t_cond(x0)
        z0 = x0  # untouched half
        # transform other half of inputs as a function of the first
        if inverse:
            s, t = -s, -t
            # what's called z1 is really x1 and vice versa since we're doing the inverse
            z1 = (x1 + t) * torch.exp(s)
        else:
            z1 = torch.exp(s) * x1 + t
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def inverse(self, z):
        return self.forward(z, inverse=True)
