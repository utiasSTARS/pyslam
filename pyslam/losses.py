import numpy as np
from numba import vectorize, float64

# Numba-vectorized is faster than boolean indexing
NUMBA_COMPILATION_TARGET = 'cpu'


class L2Loss:

    def loss(self, x):
        return 0.5 * x**2

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
        return _huber_loss(self.k, x)

    def influence(self, x):
        return _huber_infl

    def weight(self, x):
        return _huber_wght(self.k, x)


@vectorize('(float64, float64)', target=NUMBA_COMPILATION_TARGET)
def _huber_loss(k, x):
    abs_x = np.abs(x)
    if abs_x <= k:
        return 0.5 * x**2
    else:
        return k * (abs_x - 0.5 * k)


@vectorize('(float64, float64)', target=NUMBA_COMPILATION_TARGET)
def _huber_infl(k, x):
    abs_x = np.abs(x)
    if abs_x <= k:
        return x
    else:
        return k * np.sign(x)


@vectorize('(float64, float64)', target=NUMBA_COMPILATION_TARGET)
def _huber_wght(k, x):
    abs_x = np.abs(x)
    if abs_x <= k:
        return 1.
    else:
        return k / abs_x


class TukeyLoss:
    def __init__(self, k):
        self.k = k

    def loss(self, x):
        return _tukey_loss(self.k, x)

    def influence(self, x):
        return _tukey_infl(self.k, x)

    def weight(self, x):
        return _tukey_wght(self.k, x)


@vectorize('(float64, float64)', target=NUMBA_COMPILATION_TARGET)
def _tukey_loss(k, x):
    abs_x = np.abs(x)
    k_squared_over_six = k**2 / 6.
    if abs_x <= k:
        return k_squared_over_six * (1. - (1. - (x / k)**2)**3)
    else:
        return k_squared_over_six


@vectorize('(float64, float64)', target=NUMBA_COMPILATION_TARGET)
def _tukey_infl(k, x):
    abs_x = np.abs(x)
    if abs_x <= k:
        return x * (1. - (x / k)**2)
    else:
        return 0.


@vectorize('(float64, float64)', target=NUMBA_COMPILATION_TARGET)
def _tukey_wght(k, x):
    abs_x = np.abs(x)
    if abs_x <= k:
        return 1. - (x / k)**2
    else:
        return 0.
