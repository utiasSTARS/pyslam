import numpy as np
from numba import vectorize, float32, float64

# Numba-vectorized is faster than boolean indexing
NUMBA_COMPILATION_TARGET = 'parallel'


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
        infl[np.isclose(np.abs(x), 0.)] = np.nan
        return infl

    def weight(self, x):
        wght = 1. / np.abs(x)
        wght[np.isclose(np.abs(x), 0.)] = np.nan
        return wght


class CauchyLoss:

    def __init__(self, k):
        self.k = k

    def loss(self, x):
        return _cauchy_loss(self.k, x)

    def influence(self, x):
        return _cauchy_infl(self.k, x)

    def weight(self, x):
        return _cauchy_wght(self.k, x)


@vectorize([float32(float32, float32),
            float64(float64, float64)],
           nopython=True, cache=True,
           target=NUMBA_COMPILATION_TARGET)
def _cauchy_loss(k, x):
    return (0.5 * k**2) * np.log(1. + (x / k)**2)


@vectorize([float32(float32, float32),
            float64(float64, float64)],
           nopython=True, cache=True,
           target=NUMBA_COMPILATION_TARGET)
def _cauchy_infl(k, x):
    return x / (1. + (x / k)**2)


@vectorize([float32(float32, float32),
            float64(float64, float64)],
           nopython=True, cache=True,
           target=NUMBA_COMPILATION_TARGET)
def _cauchy_wght(k, x):
    return 1. / (1. + (x / k)**2)


class HuberLoss:

    def __init__(self, k):
        self.k = k

    def loss(self, x):
        return _huber_loss(self.k, x)

    def influence(self, x):
        return _huber_infl

    def weight(self, x):
        return _huber_wght(self.k, x)


@vectorize([float32(float32, float32),
            float64(float64, float64)],
           nopython=True, cache=True,
           target=NUMBA_COMPILATION_TARGET)
def _huber_loss(k, x):
    abs_x = np.abs(x)
    if abs_x <= k:
        return 0.5 * x * x
    else:
        return k * (abs_x - 0.5 * k)


@vectorize([float32(float32, float32),
            float64(float64, float64)],
           nopython=True, cache=True,
           target=NUMBA_COMPILATION_TARGET)
def _huber_infl(k, x):
    abs_x = np.abs(x)
    if abs_x <= k:
        return x
    else:
        return k * np.sign(x)


@vectorize([float32(float32, float32),
            float64(float64, float64)],
           nopython=True, cache=True,
           target=NUMBA_COMPILATION_TARGET)
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


@vectorize([float32(float32, float32),
            float64(float64, float64)],
           nopython=True, cache=True,
           target=NUMBA_COMPILATION_TARGET)
def _tukey_loss(k, x):
    abs_x = np.abs(x)
    k_squared_over_six = k**2 / 6.
    if abs_x <= k:
        return k_squared_over_six * (1. - (1. - (x / k)**2)**3)
    else:
        return k_squared_over_six


@vectorize([float32(float32, float32),
            float64(float64, float64)],
           nopython=True, cache=True,
           target=NUMBA_COMPILATION_TARGET)
def _tukey_infl(k, x):
    abs_x = np.abs(x)
    if abs_x <= k:
        return x * (1. - (x / k)**2)
    else:
        return 0.


@vectorize([float32(float32, float32),
            float64(float64, float64)],
           nopython=True, cache=True,
           target=NUMBA_COMPILATION_TARGET)
def _tukey_wght(k, x):
    abs_x = np.abs(x)
    if abs_x <= k:
        return 1. - (x / k)**2
    else:
        return 0.


class TDistributionLoss:
    def __init__(self, k):
        self.k = k
        """t-distribution degrees of freedom"""

    def loss(self, x):
        return _tdist_loss(self.k, x)

    def influence(self, x):
        return _tdist_infl(self.k, x)

    def weight(self, x):
        return _tdist_wght(self.k, x)


@vectorize([float32(float32, float32),
            float64(float64, float64)],
           nopython=True, cache=True,
           target=NUMBA_COMPILATION_TARGET)
def _tdist_loss(k, x):
    return 0.5 * (k + 1.) * np.log(1. + x * x / k)


@vectorize([float32(float32, float32),
            float64(float64, float64)],
           nopython=True, cache=True,
           target=NUMBA_COMPILATION_TARGET)
def _tdist_infl(k, x):
    return (k + 1.) * x / (k + x * x)


@vectorize([float32(float32, float32),
            float64(float64, float64)],
           nopython=True, cache=True,
           target=NUMBA_COMPILATION_TARGET)
def _tdist_wght(k, x):
    return (k + 1.) / (k + x * x)
