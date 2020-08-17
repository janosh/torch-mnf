import pickle

import numpy as np
import torch
from sklearn import datasets


class SIGGRAPH:
    """
    Taken from https://blog.evjang.com/2018/01/nf2.html.
    https://github.com/ericjang/normalizing-flows-tutorial/blob/master/siggraph.pkl
    """

    def __init__(self):
        with open("../data/siggraph.pkl", "rb") as file:
            XY = np.array(pickle.load(file), dtype="float32")
            XY -= np.mean(XY, axis=0)  # center
        self.XY = torch.from_numpy(XY)

    def sample(self, n):
        X = self.XY[np.random.randint(self.XY.shape[0], size=n)]
        return X


class Moons:
    """Two half-moons """

    def sample(self, n):
        moons = datasets.make_moons(n_samples=n, noise=0.05)[0].astype("float32")
        return torch.from_numpy(moons)


class Mixture:
    """Mixture of 4 Gaussians """

    def sample(self, n):
        assert n % 4 == 0
        r = np.r_[
            np.random.randn(n // 4, 2) * 0.5 + [0, -2],
            np.random.randn(n // 4, 2) * 0.5 + [0, 0],
            np.random.randn(n // 4, 2) * 0.5 + [2, 2],
            np.random.randn(n // 4, 2) * 0.5 + [-2, 2],
        ]
        return torch.from_numpy(r.astype("float32"))
