from torch.nn import BatchNorm1d, ReLU, Sequential

import torch_mnf.layers


# Avoid circular imports. See https://stackoverflow.com/a/42114399.
MNFLinear = lambda: torch_mnf.layers.MNFLinear


class MNFFeedForward(Sequential):
    def __init__(self, layer_sizes=[], activation=ReLU, **kwargs):
        layers = []
        for s1, s2 in zip(layer_sizes, layer_sizes[1:]):
            layers.extend(
                [MNFLinear()(s1, s2, **kwargs), activation(), BatchNorm1d(s2)]
            )
        super().__init__(*layers[:-2])  # drop final activation and batch norm

    def kl_div(self):
        """Compute current KL divergence of the whole model. Should be included
        as a regularization term in the loss function. Tensorflow will issue
        warnings "Gradients do not exist for variables of MNFLinear" if you forget.
        """
        return sum([lyr.kl_div() for lyr in self.layers if hasattr(lyr, "kl_div")])
