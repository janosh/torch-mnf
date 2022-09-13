import torch
from torch import nn

from torch_mnf import flows


class MNFLinear(nn.Module):
    """Multiplicative Normalizing Flow linear (strictly affine) layer for building
    variational Bayesian NNs.
    Reference: Christos Louizos, Max Welling (Jun 2017) https://arxiv.org/abs/1703.01961
    """

    def __init__(self, n_in, n_out, n_flows_q=2, n_flows_r=2, h_sizes=(50,)):
        """
        Args:
            n_in (int): number of input units
            n_out (int): number of output units
            n_flows_q (int, optional): length of q-flow.
            n_flows_r (int, optional): length of r-flow.
            h_sizes (list[int], optional): Number of layers and their node count
                in each hidden layer of both the q and r flow.
        """
        super().__init__()

        self.W_mean = nn.Parameter(0.1 * torch.randn([n_out, n_in]))
        self.W_log_var = nn.Parameter(-9 + 0.1 * torch.randn([n_out, n_in]))
        self.b_mean = nn.Parameter(torch.zeros(n_out))
        self.b_log_var = nn.Parameter(-9 + 0.1 * torch.randn(n_out))

        self.q0_mean = nn.Parameter(0.1 * torch.randn(n_in))
        self.q0_log_var = nn.Parameter(-9 + 0.1 * torch.randn(n_in))
        # auxiliary variable c, b1 and b2 are defined in equation (9) and (10)
        self.r0_c = nn.Parameter(0.1 * torch.randn(n_in))
        self.r0_b1 = nn.Parameter(0.1 * torch.randn(n_in))
        self.r0_b2 = nn.Parameter(0.1 * torch.randn(n_in))

        flow_q = [flows.RNVP(n_in, h_sizes=h_sizes) for _ in range(n_flows_q)]
        self.flow_q = flows.NormalizingFlow(flow_q)

        flow_r = [flows.RNVP(n_in, h_sizes=h_sizes) for _ in range(n_flows_r)]
        # flow_r parametrizes the auxiliary distribution r(z|W) used to lower bound the
        # entropy. The tightness of the bound depends on r(z|W)'s ability to approximate
        # the "auxiliary" posterior q(z|W) = q(W|z) x q(z) / q(W). Hence we use another
        # flow to improve the approximation.
        self.flow_r = flows.NormalizingFlow(flow_r)

    def forward(self, x):  # see algorithm 1 in MNF paper
        z, _ = self.sample_z(x.size(0))
        mean = x * z @ self.W_mean.T + self.b_mean

        W_var = self.W_log_var.exp()
        bias_var = self.b_log_var.exp()

        var = x**2 @ W_var.T + bias_var
        epsilon = torch.randn_like(var)

        return mean + var.sqrt() * epsilon

    def sample_z(self, batch_size=1):
        q0_std = self.q0_log_var.exp().sqrt().repeat(batch_size, 1)
        epsilon_z = torch.randn_like(q0_std)

        z = self.q0_mean + q0_std * epsilon_z
        zs, log_det_q = self.flow_q.forward(z)
        return zs[-1], log_det_q.squeeze()

    def kl_div(self):
        z, log_det_q = self.sample_z()
        W_mean = z * self.W_mean
        W_var = self.W_log_var.exp()
        epsilon_weight = torch.randn_like(W_var)
        weight = W_mean + W_var.sqrt() * epsilon_weight

        kl_div_W = 0.5 * torch.sum(-W_var.log() + W_var + W_mean**2 - 1)
        kl_div_b = 0.5 * torch.sum(
            -self.b_log_var + self.b_log_var.exp() + self.b_mean**2 - 1
        )
        log_q = -log_det_q - 0.5 * self.q0_log_var.sum()

        act = torch.tanh(self.r0_c @ weight.T)

        mean_r = self.r0_b1.ger(act).mean(1)  # eq. (9)
        log_var_r = self.r0_b2.ger(act).mean(1)  # eq. (10)

        zs, [log_det_r] = self.flow_r.forward(z)

        log_r = log_det_r + 0.5 * torch.sum(
            -log_var_r.exp() * (zs[-1] - mean_r) ** 2 + log_var_r
        )

        return kl_div_W + kl_div_b + log_q - log_r
