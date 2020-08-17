"""Improved Variational Inference with Inverse Autoregressive Flow (IAF)
Kingma et al June 2016 https://arxiv.org/abs/1606.04934

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data
and estimate densities with one forward pass only, whereas MAF would need D passes
to generate data and IAF would need D passes to estimate densities."
(MAF)
"""

import torch
from torch import nn

from torch_mnf.layers import MADE


class MAF(nn.Module):
    """Masked Autoregressive Flow that uses a MADE-style network for fast
    single-pass forward() (for density estimation) but slow dim-times-pass
    inverse() (for sampling).
    """

    def __init__(self, dim, parity, net=None, nh=24):
        super().__init__()
        self.dim = dim
        # Uses a 4-layer auto-regressive MLP by default.
        self.net = net or MADE(dim, [nh, nh, nh], 2 * dim, natural_ordering=True)
        self.parity = parity

    def forward(self, x):
        # Since we can evaluate all of z in parallel, estimation will be fast.
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        # Reverse order, so if we stack MAFs, correct things happen.
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def inverse(self, z):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0))
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone())  # clone to avoid in-place op errors if using IAF
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det


class IAF(MAF):
    """Reverses the flow of MAF, giving an Inverse Autoregressive Flow (IAF)
    with fast sampling but slow density estimation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward, self.inverse = self.inverse, self.forward
