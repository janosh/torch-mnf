from typing import Sequence

import torch
from torch import Tensor, nn


class NormalizingFlow(nn.Module):
    """A sequence of normalizing flows is a normalizing flow."""

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z: Tensor) -> Tensor:  # z -> x
        log_det = torch.zeros(z.size(0), device=z.device)
        xs = [z]
        for flow in self.flows:
            z, ld = flow.forward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det

    def inverse(self, x: Tensor) -> Tensor:  # x -> z
        log_det = torch.zeros(x.size(0), device=x.device)
        zs = [x]
        for flow in reversed(self.flows):
            x, ld = flow.inverse(x)
            log_det += ld
            zs.append(x)
        return zs, log_det


class NormalizingFlowModel(NormalizingFlow):
    """A normalizing flow model is a (base distro, flow) pair."""

    def __init__(self, base, flows) -> None:
        """Initialize a normalizing flow model."""
        super().__init__(flows)
        self.base = base

    def base_log_prob(self, x: Tensor) -> Tensor:
        """Compute the log probability of the flow's base distribution."""
        zs, _ = self.inverse(x)
        return self.base.log_prob(zs[-1])

    def sample(self, *num_samples: Sequence[int]) -> Tensor:
        """Sample from the flow's base distribution."""
        z = self.base.sample(num_samples)
        xs, _ = self.forward(z)
        return xs
