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
    im1 = np.eye(2)
    im2 = np.ones((2, 2))
    im3 = np.dstack((np.eye(2), np.ones((2, 2)), np.zeros((2, 2))))

    x = [0.5, 1]
    y = [0.5, 0]

    interp1 = pyslam.utils.bilinear_interpolate(im1, x, y)
    interp2 = pyslam.utils.bilinear_interpolate(im2, x, y)
    interp3 = pyslam.utils.bilinear_interpolate(im3, x, y)

    assert np.allclose(interp1, np.array([0.5, 0.]))
    assert np.allclose(interp2, np.array([1., 1.]))
    assert np.allclose(interp3[0, :], np.array([0.5, 1., 0.]))
    assert np.allclose(interp3[1, :], np.array([0., 1., 0.]))
