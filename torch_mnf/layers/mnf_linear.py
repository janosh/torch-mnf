import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..flows import IAF, NormalizingFlow


class MNFLinear(nn.Module):
    """Bayesian fully-connected layer with weight posterior modeled by diagonal
    covariance Gaussian. To increase expressiveness and allow for multimodality and
    non-zero covariance between weights, the Gaussian means depend on an auxiliary
    random variable z modelled by a normalizing flow. The flows base distribution is a
    standard normal.

    From "Multiplicative Normalizing Flows for Variational Bayesian Neural Networks",
    Christos Louizos, Max Welling (Jun 2017) https://arxiv.org/abs/1703.01961
    """

    def __init__(
        self,
        n_in,
        n_out,
        n_flows_q=2,
        n_flows_r=2,
        learn_p=False,
        prior_var_w=1,
        prior_var_b=1,
        flow_h_sizes=[200],
        std_init=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_in = n_in
        self.n_out = n_out

        mean_init = -9
        self.mean_W = nn.Parameter(torch.randn([n_in, self.n_out]))
        self.log_std_W = nn.Parameter(
            torch.randn([n_in, self.n_out]) * std_init + mean_init
        )
        self.mean_b = nn.Parameter(torch.zeros(self.n_out))
        self.log_var_b = nn.Parameter(torch.randn(self.n_out) * std_init + mean_init)

        # q0_mean has similar function to a dropout rate as it determines the
        # mean of the multiplicative noise z_i in eq. 4.
        self.q0_mean = nn.Parameter(torch.randn(n_in) + (0 if n_flows_q > 0 else 1))
        self.q0_log_var = nn.Parameter(torch.randn(n_in) * std_init + mean_init)

        self.r0_mean = nn.Parameter(torch.randn(n_in))
        self.r0_log_var = nn.Parameter(torch.randn(n_in))
        self.r0_apvar = nn.Parameter(torch.randn(n_in))

        self.prior_var_r_p = nn.Parameter(
            torch.randn(n_in) * std_init + np.log(prior_var_w), requires_grad=learn_p,
        )
        self.prior_var_r_p_bias = nn.Parameter(
            torch.randn(1) * std_init + np.log(prior_var_b), requires_grad=learn_p,
        )

        r_flows = [
            IAF(dim=n_in, parity=i % 2, h_sizes=flow_h_sizes) for i in range(n_flows_r)
        ]
        self.flow_r = NormalizingFlow(r_flows)

        q_flows = [
            IAF(dim=n_in, parity=i % 2, h_sizes=flow_h_sizes) for i in range(n_flows_q)
        ]
        self.flow_q = NormalizingFlow(q_flows)

    def sample_z(self, batch_size):
        q0_mean = self.q0_mean.repeat(batch_size, 1)
        epsilon = torch.randn([batch_size, self.n_in])
        q0_std = self.q0_log_var.exp().sqrt()
        z_samples = q0_mean + q0_std * epsilon

        z_samples, log_dets = self.flow_q.forward(z_samples)

        # discard intermediate transformations, only return the final result
        return z_samples[-1], log_dets

    def kl_div(self):
        z_sample, log_det_q = self.sample_z(1)

        Mtilde = z_sample.T * self.mean_W
        Vtilde = self.log_std_W.exp() ** 2
        # Stacking yields same result as outer product with ones. See eqs. 9, 10.
        iUp = torch.stack([self.prior_var_r_p.exp()] * self.n_out, dim=1)

        kl_div_w = 0.5 * torch.sum(
            iUp.log() - 2 * self.log_std_W + (Vtilde + Mtilde ** 2) / iUp - 1
        )
        kl_div_b = 0.5 * torch.sum(
            self.prior_var_r_p_bias
            - self.log_var_b
            + (self.log_var_b.exp() + self.mean_b ** 2) / self.prior_var_r_p_bias.exp()
            - 1
        )

        # Compute entropy of the initial distribution q(z_0).
        # This is independent of the actual sample z_0.
        log_q = -log_det_q.squeeze() - 0.5 * torch.sum(
            np.log(2 * np.pi) + self.q0_log_var + 1
        )

        z_sample, log_det_r = self.flow_r.forward(z_sample)

        # Shared network for hidden layer.
        mean_w = Mtilde.T @ self.r0_apvar
        var_w = Vtilde.T @ self.r0_apvar ** 2
        epsilon = torch.randn(self.n_out)
        # The bias contribution is not included in `a` since the multiplicative
        # noise is at the input units (hence it doesn't affect the biases)
        a = F.tanh(mean_w + var_w ** 0.5 * epsilon)
        # Split at output layer. Use torch.tensordot for outer product.
        mean_r = torch.tensordot(a, self.r0_mean, dims=0).mean(0)
        log_var_r = torch.tensordot(a, self.r0_log_var, dims=0).mean(0)
        # mu_tilde & sigma_tilde from eqs. 9, 10: mean and log var of the auxiliary
        # normal dist. r(z_T_b|W) from eq. 8. Used to compute first term in 15.

        log_r = log_det_r.squeeze() + 0.5 * torch.sum(
            -log_var_r.exp() * (z_sample[-1] - mean_r) ** 2
            - np.log(2 * np.pi)
            + log_var_r
        )

        return kl_div_w + kl_div_b - log_r + log_q

    def forward(self, x):
        z_samples, _ = self.sample_z(x.size(0))
        mu_out = (x * z_samples) @ self.mean_W + self.mean_b

        var_W = self.log_std_W.exp() ** 2
        var_b = self.log_var_b.exp()
        V_h = x ** 2 @ var_W + var_b
        epsilon = torch.randn(mu_out.shape)
        sigma_out = V_h ** 2 * epsilon

        return mu_out + sigma_out
