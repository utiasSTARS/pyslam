import numpy as np

import pyslam.utils


def test_invsqrt():
    sca = 4
    assert pyslam.utils.invsqrt(sca) == 1. / np.sqrt(sca)
    mat = np.array([[7,  2,  1],
                    [0,  3, -1],
                    [-3,  4, -2]])
    invsqrt_mat = pyslam.utils.invsqrt(mat)
    assert np.allclose(np.linalg.inv(np.dot(invsqrt_mat, invsqrt_mat)), mat)


def test_bilinear_interpolate():
    im = np.eye(2)
    x = [0.5, 1]
    y = [0.5, 0]
    interpolated = pyslam.utils.bilinear_interpolate(im, x, y)
    assert interpolated[0] == 0.5
    assert interpolated[1] == im[1, 0]
