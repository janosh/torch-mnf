"""
Implements various flows.
Each flow is invertible so it can be forward()ed and backward()ed.
Notice that backward() is not backward as in backprop but simply inversion.
Each flow also outputs its log abs det J^-1 "regularization" called `log_det`.

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

References:

NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516

Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770

Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data
and estimate densities with one forward pass only, whereas MAF would need D passes
to generate data and IAF would need D passes to estimate densities."
(MAF)

Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039

"Normalizing Flows for Probabilistic Modeling and Inference"
https://arxiv.org/abs/1912.02762
(review paper)
"""

import torch
from torch import nn

from nf_lib.made import MADE
from nf_lib.nets import MLP, LeafParam


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

    def backward(self, z):
        x = (z - self.t) * torch.exp(-self.s)
        log_det = torch.sum(-self.s, dim=1)
        return x, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we cleverly initialize the s, t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False

    def forward(self, x):
        # first batch is used for init
        if not self.data_dep_init_done:
            if self.scale:
                self.s.data = -x.std(dim=0, keepdim=True).log().detach()
            if self.shift:
                self.t.data = -(x * self.s.exp()).mean(dim=0, keepdim=True).detach()
            self.data_dep_init_done = True
        return super().forward(x)


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

    def backward(self, z):
        return self.forward(z, inverse=True)


class SlowMAF(nn.Module):
    """
    Masked Autoregressive Flow, slow version with explicit networks per dim
    """

    def __init__(self, dim, parity, net_class=MLP, nh=24):
        super().__init__()
        self.layers = nn.ModuleList([net_class(i, 2, nh) for i in range(1, dim)])
        self.layers.insert(0, LeafParam(2))
        self.order = list(range(dim) if parity else reversed(range(dim)))

    def forward(self, x):
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.size(0))
        for i, layer in enumerate(self.layers):
            s, t = layer(x[:, :i]).T
            z[:, self.order[i]] = x[:, i] * torch.exp(s) + t
            log_det += s
        return z, log_det

    def backward(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0))
        for i, layer in enumerate(self.layers):
            s, t = layer(x[:, :i]).T
            x[:, i] = (z[:, self.order[i]] - t) * torch.exp(-s)
            log_det += -s
        return x, log_det


class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for single-pass and
    hence fast forward() (for density estimation) but dim-times-pass, i.e. slow
    backward() (for sampling).
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

    def backward(self, z):
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
    """ Reverses the flow of MAF, giving an Inverse Autoregressive Flow (IAF)
    with fast sampling but slow density estimation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward, self.backward = self.backward, self.forward


class Invertible1x1Conv(nn.Module):
    """ As introduced in Glow paper.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P  # remains fixed during optimization
        self.L = nn.Parameter(L)  # lower triangular portion
        self.S = nn.Parameter(U.diag())  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(
            torch.triu(U, diagonal=1)
        )  # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def backward(self, z):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det


# ------------------------------------------------------------------------


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        log_det = torch.zeros(x.size(0))
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        log_det = torch.zeros(z.size(0))
        xs = [z]
        for flow in reversed(self.flows):
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (base distro, flow) pair. """

    def __init__(self, base, flows):
        super().__init__()
        self.base = base
        self.flow = NormalizingFlow(flows)

    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        base_logprob = self.base.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, log_det, base_logprob

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sample(self, *num_samples):
        z = self.base.sample(num_samples)
        xs, _ = self.flow.backward(z)
        return xs
