from torch import nn


class MLP(nn.Sequential):
    """Multilayer perceptron."""

    def __init__(self, *layer_sizes, leaky_a=0.2):
        layers = []
        for s1, s2 in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Linear(s1, s2))
            layers.append(nn.LeakyReLU(leaky_a))
        super().__init__(*layers[:-1])  # drop last ReLU
