import numpy as np
import scipy.linalg as splinalg


def invsqrt(x):
    """Convenience function to compute the inverse square root of a scalar or a square matrix."""
    if hasattr(x, 'shape'):
        if hasattr(x, 'cpu'):
            return x.__class__(
                np.linalg.inv(splinalg.sqrtm(
                    x.cpu().numpy().astype(np.float)
                ))
            )

        return np.linalg.inv(splinalg.sqrtm(x))

    return 1. / np.sqrt(x)


def bilinear_interpolate(im, x, y):
    """Perform bilinear interpolation on a 2D array. If the array is 3D, the first dimension is assumed to be colour channels.

    Args:
        im  : MxN or CxMxN array.
        x   : D-vector of x coordinates of the points at which to interpolate.
        y   : D-vector of y coordinates of the points at which to interpolate.

    Returns:
        out : D-vector or CxD vector of interpolated values
    """
    if hasattr(im, 'cpu'):
        # Got a torch tensor
        if im.dim() < 3:
            im = im.unsqueeze(dim=0)
        out = im.__class__(im.shape[0], len(x))
    else:
        # Got a numpy array
        if im.ndim < 3:
            im = np.expand_dims(im, axis=0)
        out = np.empty((im.shape[0], len(x)))

    x_min = 0
    y_min = 0
    x_max = im.shape[1] - 1
    y_max = im.shape[0] - 1

    if hasattr(x, 'floor'):
        # torch
        x0 = x.floor()
        y0 = y.floor()
    else:
        # torch
        x0 = np.floor(x)
        y0 = np.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    # Compute weights per bilinear scheme
    # Important: this needs to happen before any coordinate clipping!
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # Clip to image boundaries and sample
    # NB: at the boundaries this is equivalent to duplicating the first/last row and column
    if hasattr(x0, 'clamp_'):
        # torch
        x0.clamp_(x_min, x_max)
        x1.clamp_(x_min, x_max)
        y0.clamp_(y_min, y_max)
        y1.clamp_(y_min, y_max)
    else:
        # numpy
        x0 = np.clip(x0, x_min, x_max)
        x1 = np.clip(x1, x_min, x_max)
        y0 = np.clip(y0, y_min, y_max)
        y1 = np.clip(y1, y_min, y_max)

    # Need to cast to ints to do array access
    if hasattr(x, 'long'):
        # torch
        x0 = x0.long()
        x1 = x1.long()
        y0 = y0.long()
        y1 = y1.long()
    else:
        # numpy
        x0 = x0.astype(np.int)
        x1 = x1.astype(np.int)
        y0 = y0.astype(np.int)
        y1 = y1.astype(np.int)

    Ia = im[:, y0, x0]
    Ib = im[:, y1, x0]
    Ic = im[:, y0, x1]
    Id = im[:, y1, x1]

    # Interpolated result
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id

    if hasattr(out, 'squeeze_'):
        # torch
        out.squeeze_()
    else:
        # numpy
        out = np.squeeze(out)

    return out
