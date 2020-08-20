from torch import nn

from ..layers import MNFConv2d, MNFLinear


class MNFLeNet(nn.Sequential):
    """Bayesian LeNet with parameter posteriors modeled by normalizing flows."""

    def __init__(self, **kwargs):
        layers = [
            MNFConv2d(1, 20, kernel_size=5, **kwargs),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            MNFConv2d(20, 50, kernel_size=5, **kwargs),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            MNFLinear(50 * 16, 50, **kwargs),
            nn.ReLU(),
            MNFLinear(50, 10, **kwargs),
            nn.LogSoftmax(dim=-1),
        ]
        super().__init__(*layers)

    # just for debugging, this is what happens anyway in nn.Sequential
    def forward(self, x):
        for lyr in self:
            x = lyr(x)
        return x

    def kl_div(self):
        """Compute current KL divergence of the whole model. Given by the sum
        of KL divs. from each MNF layer. Use as a regularization term during training.
        """
        return sum([lyr.kl_div() for lyr in self if hasattr(lyr, "kl_div")])
