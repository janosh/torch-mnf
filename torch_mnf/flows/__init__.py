"""
Implements various flows.
Each flow is invertible and outputs its log abs det J^-1 "regularization"
via the log_det method.

The Jacobian's determinant measures the local change of volume (due to
stretching or squashing of space) under the flows transformation function
y = f(x). We take the inverse Jacobian because we want to know how volume
changes under x = f^-1(y), going from the space Y of real-world data we can
observe to the space X of the base distribution. We work in log space for
numerical stability, and take the absolute value because we don't care about
changes in orientation, i.e. whether space is mirrored/reflected. We only
care if it is stretched, since all we need is conservation of probability
mass to retain a valid transformed PDF under f, no matter where in space
that mass ends up.
"""

import torch
from torch import nn

# from torch_mnf.flows.act_norm import ActNorm  # noqa
from torch_mnf.flows.affine_constant_flow import AffineConstantFlow  # noqa
from torch_mnf.flows.affine_half_flow import AffineHalfFlow  # noqa
from torch_mnf.flows.glow import Glow  # noqa
from torch_mnf.flows.maf import MAF  # noqa
from torch_mnf.flows.spline import NSF_AR, NSF_CL  # noqa


class NormalizingFlow(nn.Module):
    """A sequence of Normalizing Flows is a Normalizing Flow"""

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):  # z -> x
        log_det = torch.zeros(x.size(0))
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def inverse(self, z):  # x -> z
        log_det = torch.zeros(z.size(0))
        xs = [z]
        for flow in reversed(self.flows):
            z, ld = flow.inverse(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """A Normalizing Flow Model is a (base distro, flow) pair."""

    def __init__(self, base, flows):
        super().__init__()
        self.base = base
        self.flow = NormalizingFlow(flows)

    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        base_logprob = self.base.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, log_det, base_logprob

    def inverse(self, z):
        xs, log_det = self.flow.inverse(z)
        return xs, log_det

    def sample(self, *num_samples):
        z = self.base.sample(num_samples)
        xs, _ = self.flow.inverse(z)
        return xs
