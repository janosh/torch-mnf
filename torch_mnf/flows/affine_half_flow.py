"""NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516.

Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770

Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from torch_mnf.models import MLP


class AffineHalfFlow(nn.Module):
    """As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half
    of dimensions in x are linearly scaled/transformed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts.
    """

    def __init__(
        self,
        dim: int,
        parity: bool,
        h_sizes: Sequence[int] = (24, 24, 24),
        scale: bool = True,
        shift: bool = True,
    ) -> None:
        super().__init__()
        self.parity = parity
        self.s_net = self.t_net = lambda x: x.new_zeros(x.size(0), dim // 2)
        if scale:
            self.s_net = MLP(dim // 2, *h_sizes, dim // 2)
        if shift:
            self.t_net = MLP(dim // 2, *h_sizes, dim // 2)

    def forward(self, z: Tensor, inverse: bool = False) -> tuple[Tensor, Tensor]:
        """Affine half flow forward pass."""
        z0, z1 = z.chunk(2, dim=1)
        if self.parity:
            z0, z1 = z1, z0
        s, t = self.s_net(z0), self.t_net(z0)
        x0 = z0  # untouched half
        # transform z1 as a function of z0
        if inverse:
            x1 = (z1 - t) / s.exp()
            # what's called x1 is really z1 and vice versa since we're doing the inverse
            s = -s  # change sign of s to get the right log_det below
        else:
            x1 = s.exp() * z1 + t
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = s.sum(1)
        return x, log_det

    def inverse(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Affine half flow inverse pass."""
        return self.forward(x, inverse=True)
