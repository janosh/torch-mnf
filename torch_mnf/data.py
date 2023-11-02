import pickle

import numpy as np
import torch
from sklearn.datasets import make_blobs, make_moons

from .utils import ROOT


def sample_siggraph(n_samples: int) -> torch.Tensor:
    """Taken from https://blog.evjang.com/2018/01/nf2.html.
    https://github.com/ericjang/normalizing-flows-tutorial/blob/master/siggraph.pkl.
    """
    with open(ROOT + "/data/siggraph.pkl", "rb") as file:
        XY = np.array(pickle.load(file))
    XY -= XY.mean(0)  # center
    XY = torch.as_tensor(XY).float()
    return XY[np.random.randint(XY.shape[0], size=n_samples)]


def sample_moons(n_samples: int) -> torch.Tensor:
    """Probability density forming two interwoven half-moons."""
    samples, _ = make_moons(n_samples, noise=0.05, random_state=0)
    return torch.as_tensor(samples).float()


def sample_blobs(n_samples: int) -> torch.Tensor:
    """Mixture of 3 Gaussians."""
    centers = [(3, 3), (0, 0), (-3, -3)]
    samples, _ = make_blobs(n_samples, centers=centers, random_state=0)
    return torch.as_tensor(samples).float()
