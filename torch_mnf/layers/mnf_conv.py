import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..flows import IAF, NormalizingFlow


class MNFConv2d(nn.Module):
    """Bayesian 2D convolutional layer with weight posterior modeled by diagonal
    covariance Gaussian. To increase expressiveness and allow for multimodality and
    non-zero covariance between weights, the Gaussian means depend on an auxiliary
    random variable z modelled by a normalizing flow. The flows base distribution is a
    standard normal.

    From "Multiplicative Normalizing Flows for Variational Bayesian Neural Networks",
    Christos Louizos, Max Welling (Jun 2017) https://arxiv.org/abs/1703.01961
    """

    def __init__(
        self,
        in_channels,  # = 1 for black & white images like MNIST
        out_channels,
        kernel_size,  # int for kernel width and height
        n_flows_q=1,
        n_flows_r=1,
        learn_p=False,
        use_z=True,
        prior_var_w=1,
        prior_var_b=1,
        flow_h_sizes=[200],
        max_std=1,
        std_init=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_std = max_std
        self.use_z = use_z

        n_rows = n_cols = kernel_size
        self.input_dim = in_channels * n_cols * n_rows

        mean_init = -9
        W_shape = (out_channels, in_channels, n_rows, n_cols)
        self.mean_W = nn.Parameter(torch.randn(W_shape))
        self.log_std_W = nn.Parameter(torch.randn(W_shape) * std_init + mean_init)
        self.mean_b = nn.Parameter(torch.zeros(out_channels))
        self.log_var_b = nn.Parameter(torch.randn(out_channels) * std_init + mean_init)

        if self.use_z:
            # q0_mean has similar function to a dropout rate as it determines the
            # mean of the multiplicative noise z_k in eq. 5.
            self.q0_mean = nn.Parameter(
                torch.randn(out_channels) + (0 if n_flows_q > 0 else 1)
            )
            self.q0_log_var = nn.Parameter(
                torch.randn(out_channels) * std_init + mean_init
            )

            self.r0_mean = nn.Parameter(torch.randn(out_channels))
            self.r0_log_var = nn.Parameter(torch.randn(out_channels))
            self.r0_apvar = nn.Parameter(torch.randn(out_channels))

        self.prior_var_r_p = nn.Parameter(
            torch.randn(self.input_dim) * std_init + np.log(prior_var_w),
            requires_grad=learn_p,
        )
        self.prior_var_r_p_bias = nn.Parameter(
            torch.randn([]) * std_init + np.log(prior_var_b), requires_grad=learn_p,
        )

        r_flows = [
            IAF(dim=out_channels, parity=i % 2, h_sizes=flow_h_sizes)
            for i in range(n_flows_r)
        ]
        self.flow_r = NormalizingFlow(r_flows)

        q_flows = [
            IAF(dim=out_channels, parity=i % 2, h_sizes=flow_h_sizes)
            for i in range(n_flows_q)
        ]
        self.flow_q = NormalizingFlow(q_flows)

    def forward(self, x):
        z_samples, _ = self.sample_z(x.size(0))
        mean, var = self.get_mean_var(x)

        mu_out = mean * z_samples[..., None, None]  # add singleton dims
        epsilon = torch.randn(mu_out.shape)
        sigma_out = var.sqrt() * epsilon

        return mu_out + sigma_out

    def sample_z(self, batch_size):
        log_dets = torch.zeros(batch_size)
        if not self.use_z:
            return torch.ones([batch_size, self.out_channels]), log_dets

        q0_mean = self.q0_mean.repeat(batch_size, 1)
        epsilon = torch.randn(batch_size, self.out_channels)
        q0_std = self.q0_log_var.exp().sqrt()
        z_samples = q0_mean + q0_std * epsilon

        z_samples, log_dets = self.flow_q.forward(z_samples)

        return z_samples[-1], log_dets

    def get_mean_var(self, x):
        var_w = torch.clamp(self.log_std_W.exp(), 0, self.max_std) ** 2
        var_b = torch.clamp(self.log_var_b.exp(), 0, self.max_std ** 2)

        # Perform cross-correlation.
        mean_W_out = F.conv2d(x, weight=self.mean_W)
        var_w_out = F.conv2d(x ** 2, weight=var_w)
        return mean_W_out + self.mean_b.view(-1, 1, 1), var_w_out + var_b.view(-1, 1, 1)

    def kl_div(self):
        z_sample, log_det_q = self.sample_z(1)

        std_w = torch.exp(self.log_std_W)
        std_w = std_w.reshape(-1, self.out_channels)
        mu_w = self.mean_W.reshape(-1, self.out_channels)
        Mtilde = mu_w * z_sample
        mean_b = self.mean_b * z_sample
        Vtilde = std_w ** 2
        # Stacking yields same result as outer product with ones. See eqs. 11, 12.
        iUp = torch.stack([torch.exp(self.prior_var_r_p)] * self.out_channels, dim=1)

        kl_div_w = 0.5 * torch.sum(
            np.log(iUp) - std_w.log() + (Vtilde + Mtilde ** 2) / iUp - 1
        )
        kl_div_b = 0.5 * torch.sum(
            self.prior_var_r_p_bias
            - self.log_var_b
            + (torch.exp(self.log_var_b) + mean_b ** 2)
            / torch.exp(self.prior_var_r_p_bias)
            - 1
        )

        log_q = -log_det_q.squeeze()
        if self.use_z:
            # Compute entropy of the initial distribution q(z_0).
            # This is independent of the actual sample z_0.
            log_q -= 0.5 * torch.sum(np.log(2 * np.pi) + self.q0_log_var + 1)

        log_r = 0
        if self.use_z:
            z_sample, log_det_r = self.flow_r.forward(z_sample)
            log_r = log_det_r.squeeze()

            mean_w = Mtilde @ self.r0_apvar
            var_w = Vtilde @ self.r0_apvar ** 2
            epsilon = torch.randn(self.input_dim)
            # For convolutional layers, linear mappings empirically work better than
            # tanh non-linearity. Hence the removal of a = tf.tanh(a). Christos Louizos
            # confirmed this in https://github.com/AMLab-Amsterdam/MNF_VBNN/issues/4
            # even though the paper states the use of tanh in conv layers.
            a = mean_w + var_w.sqrt() * epsilon
            # a = tf.tanh(a)
            mu_b = torch.sum(mean_b * self.r0_apvar)
            var_b = torch.sum(torch.exp(self.log_var_b) * self.r0_apvar ** 2)
            a += mu_b + var_b.sqrt() * torch.randn([])
            # a = tf.tanh(a)

            # Mean and log variance of the auxiliary normal dist. r(z_T_b|W) in eq. 8.
            mean_r = torch.tensordot(a, self.r0_mean, dims=0).mean(0)
            log_var_r = torch.tensordot(a, self.r0_log_var, dims=0).mean(0)

            # Log likelihood of a zero-covariance normal dist: ln N(x | mu, sigma) =
            # -1/2 sum_dims(ln(2 pi) + ln(sigma^2) + (x - mu)^2 / sigma^2)
            log_r += 0.5 * torch.sum(
                -torch.exp(log_var_r) * (z_sample[-1] - mean_r) ** 2
                - np.log(2 * np.pi)
                + log_var_r
            )

        return kl_div_w + kl_div_b - log_r + log_q
