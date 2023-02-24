"""Implements a Masked Autoregressive MLP, where carefully constructed
binary masks over weights ensure the autoregressive property.
Copied from https://github.com/karpathy/pytorch-made.
"""

import numpy as np
import torch
from torch import nn


class MaskedLinear(nn.Linear):
    """A dense layer with a configurable mask on the weights."""

    def __init__(self, n_in, n_out, bias=True):
        super().__init__(n_in, n_out, bias)
        self.register_buffer("mask", torch.ones(n_in, n_out))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask))

    def forward(self, x):
        return x @ (self.weight.T * self.mask) + self.bias


class MADE(nn.Sequential):
    def __init__(self, n_in, hidden_sizes, n_out, num_masks=1, natural_ordering=False):
        """n_in (int): number of inputs
        hidden sizes (list of ints): number of units in hidden layers
        n_out (int): number of outputs, which usually collectively parameterize some 1D
            distribution. Note: if n_out is e.g. 2x larger than n_in (perhaps the mean
            and std), then the first n_in will be all the means and the second n_in will
            be the std. devs. Make sure to deconstruct this correctly downstream.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: retain ordering of inputs, don't use random permutations.
        """
        assert n_out % n_in == 0, "n_out must be integer multiple of n_in"
        self.n_in = n_in
        self.n_out = n_out
        self.hidden_sizes = hidden_sizes

        # define a simple MLP neural net
        layers = []
        hs = [n_in] + list(hidden_sizes) + [n_out]
        for h0, h1 in zip(hs, hs[1:]):
            layers.extend([MaskedLinear(h0, h1), nn.ReLU()])
        super().__init__(*layers[:-1])  # drop last ReLU)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also pre-compute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return  # only a single seed, skip for efficiency

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = (
            np.arange(self.n_in)
            if self.natural_ordering
            else rng.permutation(self.n_in)
        )
        for lyr, size in enumerate(self.hidden_sizes):
            # Use minimum connectivity of previous layer as lower bound when sampling
            # values for m_l(k) to avoid unconnected units. See comment after eq. (13).
            self.m[lyr] = rng.randint(self.m[lyr - 1].min(), self.n_in - 1, size=size)

        # construct the mask matrices
        n_layers = len(self.hidden_sizes)
        masks = [
            self.m[lyr - 1][:, None] <= self.m[lyr][None, :] for lyr in range(n_layers)
        ]
        masks.append(self.m[n_layers - 1][:, None] < self.m[-1][None, :])

        # handle the case where n_out = n_in * k, for integer k > 1
        if self.n_out > self.n_in:
            k = int(self.n_out / self.n_in)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        masked_layers = [lyr for lyr in self if isinstance(lyr, MaskedLinear)]
        for lyr, m in zip(masked_layers, masks):
            lyr.set_mask(m)
