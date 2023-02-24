import torch
from torch import nn


class Glow(nn.Module):
    """Glow: Generative Flow with Invertible 1x1 Convolutions.
    Kingma and Dhariwal, Jul 2018, https://arxiv.org/abs/1807.03039.
    """

    def __init__(self, dim):
        super().__init__()
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P  # remains fixed during optimization
        self.L = nn.Parameter(L)  # lower triangular portion
        self.S = nn.Parameter(U.diag())  # "crop out" the diagonal to its own parameter
        # "crop out" diagonal, stored in S
        self.U = nn.Parameter(torch.triu(U, diagonal=1))

    def _assemble_W(self):
        """Assemble W from its pieces (P, L, U, S)."""
        L = torch.tril(self.L, diagonal=-1) + torch.eye(len(self.L))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + self.S.diag())
        return W

    def forward(self, z):
        W = self._assemble_W()
        x = z @ W
        log_det = self.S.abs().log().sum()
        return x, log_det

    def inverse(self, x):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        z = x @ W_inv
        log_det = -self.S.abs().log().sum()
        return z, log_det
