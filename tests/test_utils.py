import numpy as np

import pyslam.utils


def test_bilinear_interpolate():
    im = np.eye(2)
    x = [0.5, 1]
    y = [0.5, 0]
    interpolated = pyslam.utils.bilinear_interpolate(im, x, y)
    assert interpolated[0] == 0.5
    assert interpolated[1] == im[1, 0]
