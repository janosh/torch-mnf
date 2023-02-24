from typing import Any

from torch import nn

import torch_mnf.layers


class MNFLeNet(nn.Sequential):
    """Bayesian LeNet with parameter posteriors modeled by normalizing flows."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model."""
        layers = [
            torch_mnf.layers.MNFConv2d(1, 20, kernel_size=5, **kwargs),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            torch_mnf.layers.MNFConv2d(20, 50, kernel_size=5, **kwargs),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            torch_mnf.layers.MNFLinear(50 * 16, 50, **kwargs),
            nn.ReLU(),
            torch_mnf.layers.MNFLinear(50, 10, **kwargs),
            nn.LogSoftmax(dim=-1),
        ]
        super().__init__(*layers)

    def kl_div(self) -> float:
        """Compute current KL divergence of the whole model. Given by the sum
        of KL divs. from each MNF layer. Use as a regularization term during training.
        """
        return sum(lyr.kl_div() for lyr in self if hasattr(lyr, "kl_div"))
