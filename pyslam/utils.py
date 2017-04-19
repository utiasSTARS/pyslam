import numpy as np
import scipy.linalg as splinalg


def invsqrt(x):
    """Convenience function to compute the inverse square root of a scalar or a square matrix."""
    if hasattr(x, 'shape'):
        return np.linalg.inv(splinalg.sqrtm(x))

    return 1. / np.sqrt(x)
