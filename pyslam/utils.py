import numpy as np


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if not x.shape == y.shape:
        raise ValueError("x and y must have the same shape")

    interpolated = np.empty(x.shape)
    interpolated.fill(np.nan)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # x0 = np.clip(x0, 0, im.shape[1] - 1)
    # x1 = np.clip(x1, 0, im.shape[1] - 1)
    # y0 = np.clip(y0, 0, im.shape[0] - 1)
    # y1 = np.clip(y1, 0, im.shape[0] - 1)

    valid_mask = (x0 >= 0) &  (x0 < im.shape[1]) & (x1 >= 0) & (x1 < im.shape[1]) \
        & (y0 >= 0) & (y0 < im.shape[0]) & (y1 >= 0) & (y1 < im.shape[0])

    valid_x = x[valid_mask]
    valid_x0 = x0[valid_mask]
    valid_x1 = x1[valid_mask]
    valid_y = y[valid_mask]
    valid_y0 = y0[valid_mask]
    valid_y1 = y1[valid_mask]

    Ia = im[valid_y0, valid_x0]
    Ib = im[valid_y1, valid_x0]
    Ic = im[valid_y0, valid_x1]
    Id = im[valid_y1, valid_x1]

    wa = (valid_x1 - valid_x) * (valid_y1 - valid_y)
    wb = (valid_x1 - valid_x) * (valid_y - valid_y0)
    wc = (valid_x - valid_x0) * (valid_y1 - valid_y)
    wd = (valid_x - valid_x0) * (valid_y - valid_y0)

    interpolated[valid_mask] = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return interpolated
