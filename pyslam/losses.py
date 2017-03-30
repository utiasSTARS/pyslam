import numpy as np


class L2Loss:

    def loss(self, x):
        return 0.5 * x * x

    def influence(self, x):
        return x

    def weight(self, x):
        return np.ones(x.size)


class L1Loss:

    def loss(self, x):
        return np.abs(x)

    def influence(self, x):
        infl = np.sign(x)
        infl[np.abs(x) < 0.1] = np.nan
        return infl

    def weight(self, x):
        wght = 1. / np.abs(x)
        wght[np.abs(x) < 0.1] = np.nan
        return wght


class CauchyLoss:

    def __init__(self, k):
        self.k = k

    def loss(self, x):
        return (0.5 * self.k**2) * np.log(1 + (x / self.k)**2)

    def influence(self, x):
        return x / (1 + (x / self.k)**2)

    def weight(self, x):
        return 1 / (1 + (x / self.k)**2)


class HuberLoss:

    def __init__(self, k):
        self.k = k

    def loss(self, x):
        loss = np.zeros(x.size)
        leq_mask = np.abs(x) <= self.k
        ge_mask = np.abs(x) > self.k
        loss[leq_mask] = 0.5 * x[leq_mask]**2
        loss[ge_mask] = self.k * (np.abs(x[ge_mask]) - 0.5 * self.k)
        return loss

    def influence(self, x):
        infl = np.zeros(x.size)
        leq_mask = np.abs(x) <= self.k
        ge_mask = np.abs(x) > self.k
        infl[leq_mask] = x[leq_mask]
        infl[ge_mask] = self.k * np.sign(x[ge_mask])
        return infl

    def weight(self, x):
        wght = np.zeros(x.size)
        leq_mask = np.abs(x) <= self.k
        ge_mask = np.abs(x) > self.k
        wght[leq_mask] = np.ones(x[leq_mask].size)
        wght[ge_mask] = self.k / np.abs(x[ge_mask])
        return wght
