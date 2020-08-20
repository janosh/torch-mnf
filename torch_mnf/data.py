import os
import pickle

import numpy as np
import torch
from sklearn.datasets import make_moons

ROOT = os.getcwd().split("/torch_mnf", 1)[0]


def sample_siggraph(n_samples):
    """Taken from https://blog.evjang.com/2018/01/nf2.html.
    https://github.com/ericjang/normalizing-flows-tutorial/blob/master/siggraph.pkl
    """
    with open(ROOT + "/data/siggraph.pkl", "rb") as file:
        XY = np.array(pickle.load(file), dtype="float32")
        XY -= np.mean(XY, axis=0)  # center
    XY = torch.from_numpy(XY)
    X = XY[np.random.randint(XY.shape[0], size=n_samples)]
    return X


def sample_moons(n_samples):
    """Probability density forming two interwoven half-moons."""
    samples = make_moons(n_samples, noise=0.05, random_state=0)[0].astype("float32")
    return torch.from_numpy(samples)


def sample_gaussian_mixture(n_samples):
    """Mixture of 4 Gaussians."""
    assert n_samples % 4 == 0
    r = np.r_[
        np.random.randn(n_samples // 4, 2) * 0.5 + [0, -2],
        np.random.randn(n_samples // 4, 2) * 0.5 + [0, 0],
        np.random.randn(n_samples // 4, 2) * 0.5 + [2, 2],
        np.random.randn(n_samples // 4, 2) * 0.5 + [-2, 2],
    ]
    return torch.from_numpy(r.astype("float32"))
