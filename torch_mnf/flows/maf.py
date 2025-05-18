"""Improved Variational Inference with Inverse Autoregressive Flow (IAF)
Kingma et al June 2016 https://arxiv.org/abs/1606.04934.

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data
and estimate densities with one forward pass only, whereas MAF would need D passes
to generate data and IAF would need D passes to estimate densities."
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from torch_mnf.layers import MADE


class MAF(nn.Module):
    """Masked Autoregressive Flow that uses a MADE-style network for fast
    single-pass forward() (for density estimation) but slow dim-times-pass
    inverse() (for sampling).
    """

    def __init__(
        self,
        dim: int,
        parity: bool,
        net: nn.Module | None = None,
        h_sizes: Sequence[int] = (24, 24, 24),
    ) -> None:
        super().__init__()
        # Uses a 4-layer auto-regressive MLP by default.
        self.net = net or MADE(dim, h_sizes, 2 * dim, natural_ordering=True)
        self.parity = parity

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MAF forward pass."""
        batch_size, z_size = z.shape
        # we have to decode elements of x sequentially one at a time
        x = torch.zeros_like(z)
        log_det = torch.zeros(batch_size)
        z = z.flip(dims=[1]) if self.parity else z
        for i in range(z_size):
            st = self.net(x.clone())  # clone to avoid in-place op errors if using IAF
            s, t = st.split(z_size, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MAF inverse pass."""
        # Since we can evaluate all of z in parallel, density estimation is fast.
        st = self.net(x)
        s, t = st.split(x.size(1), dim=1)
        z = x * s.exp() + t
        # Reverse order, so if we stack MAFs, correct things happen.
        z = z.flip(dims=[1]) if self.parity else z
        log_det = s.sum(1)
        return z, log_det


class IAF(MAF):
    """Reverses the flow of MAF, giving an Inverse Autoregressive Flow (IAF)
    with fast sampling but slow density estimation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward, self.inverse = self.inverse, self.forward  # type: ignore[assignment]
