from torch import nn


class MLP(nn.Module):
    """Just a 4-layer perceptron. """

    def __init__(self, n_in, n_out, n_h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_h),
            nn.LeakyReLU(0.2),
            nn.Linear(n_h, n_h),
            nn.LeakyReLU(0.2),
            nn.Linear(n_h, n_h),
            nn.LeakyReLU(0.2),
            nn.Linear(n_h, n_out),
        )

    def forward(self, x):
        return self.net(x)
