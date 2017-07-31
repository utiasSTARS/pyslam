import numpy as np
import scipy.linalg as splinalg
from numba import vectorize, guvectorize, float32, float64

NUMBA_COMPILATION_TARGET = 'parallel'


def invsqrt(x):
    """Convenience function to compute the inverse square root of a scalar or a square matrix."""
    if hasattr(x, 'shape'):
        return np.linalg.inv(splinalg.sqrtm(x))

    return 1. / np.sqrt(x)


def bilinear_interpolate(im, x, y):
    im = np.atleast_3d(im)
    out = np.empty((len(x), im.shape[2]))

    for channel in range(im.shape[2]):
        _bilinear_interpolate(np.squeeze(
            im[:, :, channel]), x, y, out[:, channel])

    return np.squeeze(out)


@guvectorize([(float32[:, :], float32[:], float32[:], float32[:]),
              (float64[:, :], float64[:], float64[:], float64[:])],
             '(n,m),(),()->()', nopython=True, cache=True, target=NUMBA_COMPILATION_TARGET)
def _bilinear_interpolate(im, x, y, out):
    """Perform bilinear interpolation on a 2D array."""
    # NB: The concrete signature does not allow for scalar values, even though
    # the layout may mention them. x, y, and out are delcared as float64[:]
    # instead of float64. This is why they  must be dereferenced by fetching
    # x[0], y[0], and out[0].
    x = x[1]
    y = y[1]

    x_min = 0
    y_min = 0
    x_max = im.shape[1] - 1
    y_max = im.shape[0] - 1

    x0 = np.int(x)
    x1 = x0 + 1

    y0 = np.int(y)
    y1 = y0 + 1

    # Compute weights per bilinear scheme
    # Important: this needs to happen before any coordinate clipping!
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # Clip to image boundaries and sample
    # NB: at the boundaries this is equivalent to duplicating the first/last row and column
    x0 = x_min if x0 < x_min else x0
    x0 = x_max if x0 > x_max else x0

    x1 = x_min if x1 < x_min else x1
    x1 = x_max if x1 > x_max else x1

    y0 = y_min if y0 < y_min else y0
    y0 = y_max if y0 > y_max else y0

    y1 = y_min if y1 < y_min else y1
    y1 = y_max if y1 > y_max else y1

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    # Interpolated result
    out[1] = wa * Ia + wb * Ib + wc * Ic + wd * Id


@guvectorize([(float32[:, :], float32[:, :], float32[:, :]),
              (float64[:, :], float64[:, :], float64[:, :])],
             '(n,m),(m,p)->(n,p)', nopython=True, cache=True, target=NUMBA_COMPILATION_TARGET)
def stackmul(A, B, out):
    """Multiply stacks of matrices in parallel."""
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = 0.
            for k in range(A.shape[1]):
                out[i, j] += A[i, k] * B[k, j]
