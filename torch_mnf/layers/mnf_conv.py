from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from torch_mnf import flows


class MNFConv2d(nn.Module):
    """Bayesian 2D convolutional layer with weight posterior modeled by diagonal
    covariance Gaussian. To increase expressiveness and allow for multimodality and
    non-zero covariance between weights, the Gaussian means depend on an auxiliary
    random variable z modelled by a normalizing flow with Gaussian base distribution.

    From "Multiplicative Normalizing Flows for Variational Bayesian Neural Networks",
    Christos Louizos, Max Welling (Jun 2017) https://arxiv.org/abs/1703.01961
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int,
        n_flows_q: int = 2,
        n_flows_r: int = 2,
        h_sizes: Sequence[int] = (50,),
    ) -> None:
        """Args:
        n_in (int): number of input channels
        n_out (int): number of output channels
        kernel_size (int): side length of square kernel
        n_flows_q (int, optional): length of q-flow.
        n_flows_r (int, optional): length of r-flow.
        h_sizes (list[int], optional): Number of layers and their node count
        in each hidden layer of both the q and r flow.
        """
        super().__init__()

        # weight shape: in_channels, out_channels, kernel_rows, kernel_cols
        W_shape = [n_out, n_in, kernel_size, kernel_size]

        self.W_mean = nn.Parameter(0.1 * torch.randn(W_shape))
        self.W_log_var = nn.Parameter(-9 + 0.1 * torch.randn(W_shape))
        self.b_mean = torch.zeros(n_out)
        self.b_log_var = nn.Parameter(-9 + 0.1 * torch.randn(n_out))

        # q_0(z) is the flow's base distribution for the auxiliary random variable z
        # that's used to increase expressivity of the network weight posterior q(W|z).
        self.q0_mean = nn.Parameter(0.1 * torch.randn(n_out))
        # q0_mean has a similar function to a dropout rate as it determines the mean of
        # the multiplicative noise z_k in eq. 5 of the MNF paper.
        self.q0_log_var = nn.Parameter(-9 + 0.1 * torch.randn(n_out))
        # auxiliary variables c, b1, b2 defined in eqs. (11), (12)
        self.r0_c = nn.Parameter(0.1 * torch.randn(n_out))
        self.r0_b1 = nn.Parameter(0.1 * torch.randn(n_out))
        self.r0_b2 = nn.Parameter(0.1 * torch.randn(n_out))

        # flows.AffineHalfFlow(n_out, parity=i % 2, h_sizes=h_sizes)
        q_flows = [flows.RNVP(n_out, h_sizes=h_sizes) for _ in range(n_flows_q)]
        self.flow_q = flows.NormalizingFlow(q_flows)

        # flows.AffineHalfFlow(n_out, parity=i % 2, h_sizes=h_sizes)
        r_flows = [flows.RNVP(n_out, h_sizes=h_sizes) for _ in range(n_flows_r)]
        self.flow_r = flows.NormalizingFlow(r_flows)

    def forward(self, x):  # see algorithm 2 in MNF paper
        z, _ = self.sample_z()
        W_var = self.W_log_var.exp()
        b_var = self.b_log_var.exp()

        W_mean = self.W_mean * z.view(-1, 1, 1, 1)

        mean = F.conv2d(x, weight=W_mean, bias=self.b_mean)
        var = F.conv2d(x**2, weight=W_var, bias=b_var)
        epsilon = torch.randn_like(var)

        return mean + var.sqrt() * epsilon

    def sample_z(self):
        q0_std = self.q0_log_var.exp().sqrt()
        epsilon_z = torch.randn_like(q0_std)
        z = self.q0_mean + q0_std * epsilon_z

        zs, log_dets = self.flow_q.forward(z[None, ...])

        # discard intermediate flow transformations, only return the final RVs
        return zs[-1], log_dets.squeeze()

    def kl_div(self):
        z, log_det_q = self.sample_z()

        W_var = self.W_log_var.exp()
        b_var = self.b_log_var.exp()
        W_mean = self.W_mean * z.view(-1, 1, 1, 1)
        b_mean = self.b_mean * z

        kl_div_W = 0.5 * torch.sum(-W_var.log() + W_var + W_mean**2 - 1)
        kl_div_b = 0.5 * torch.sum(-b_var.log() + b_var + b_mean**2 - 1)

        # log_q_z0 = entropy of the initial distribution q(z_0). For a Gaussian, this is
        # 1/2 ln(2 pi e sigma^2) but we drop the constant offset of 1/2 ln(2 pi e) and
        # just compute ln(sigma).
        log_q_z0 = 0.5 * self.q0_log_var.sum()
        log_q = -log_det_q - log_q_z0

        W_mean = W_mean.view(-1, len(self.r0_c)) @ self.r0_c  # eq. (11)
        W_std = W_var.sqrt().view(-1, len(self.r0_c)) @ self.r0_c  # eq. (12)
        epsilon_w = torch.randn_like(W_std)
        # For convolutional layers, linear mappings empirically work better than
        # tanh. Hence no need for act = tanh(act). Christos Louizos
        # confirmed this in https://github.com/AMLab-Amsterdam/MNF_VBNN/issues/4
        # even though the paper states the use of tanh in conv layers.
        act = W_mean + W_std * epsilon_w

        b_mean = torch.sum(b_mean * self.r0_c)
        b_var = torch.sum(self.b_log_var.exp() * self.r0_c**2)
        epsilon_b = torch.randn([])
        act += b_mean + b_var.sqrt() * epsilon_b

        # Mean and log variance of the auxiliary normal dist. r(z_T_b|W) in eq. 8.
        mean_r = self.r0_b1.ger(act).mean(1)
        log_var_r = self.r0_b2.ger(act).mean(1)

        zs, [log_det_r] = self.flow_r.forward(z)

        # Log likelihood of a zero-covariance normal dist: ln N(x | mu, sigma) =
        # -1/2 sum_dims(ln(2 pi) + ln(sigma^2) + (x - mu)^2 / sigma^2)
        log_r = log_det_r + 0.5 * torch.sum(
            -log_var_r.exp() * (zs[-1] - mean_r) ** 2 + log_var_r
        )

        return kl_div_W + kl_div_b + log_q - log_r
