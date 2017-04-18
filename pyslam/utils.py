import numpy as np
import scipy.linalg as splinalg
from numba import guvectorize, int32, int64, float32, float64


def invsqrt(x):
    """Convenience function to compute the inverse square root of a scalar or a square matrix."""
    if hasattr(x, 'shape'):
        return np.linalg.inv(splinalg.sqrtm(x))

    return 1. / np.sqrt(x)


def bilinear_interpolate(im, x, y):
    """Perform bilinear interpolation on a 2D array."""
    x = np.asarray(x)
    y = np.asarray(y)

    if not x.shape == y.shape:
        raise ValueError("x and y must have the same shape")

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


@guvectorize([(float64[:, :], float64[:, :], float64[:, :])],
             '(n,m),(m,p)->(n,p)', nopython=True, cache=True, target='cpu')
def stackmul(A, B, out):
    """Multiply two stacks of matrices in parallel."""
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = 0.
            for k in range(A.shape[1]):
                out[i, j] += A[i, k] * B[k, j]
