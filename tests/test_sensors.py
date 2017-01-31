import pytest
import numpy as np


class TestStereoCamera:

    @pytest.fixture
    def sensor(self):
        from pyslam.sensors import StereoCamera
        cu = 100.
        cv = 100.
        fu = 200.
        fv = 200.
        b = 1.
        w = 200
        h = 200
        return StereoCamera(cu, cv, fu, fv, b, w, h)

    def test_is_valid_measurement(self, sensor):
        test_uvd = np.array([[110., 120., 10.],
                             [-10., 100., 10.],
                             [0., -10., 10.],
                             [0., 0., -5.]])
        test_expected = [True, False, False, False]
        for uvd, expected in zip(test_uvd, test_expected):
            assert sensor.is_valid_measurement(uvd) == expected

        assert np.array_equal(
            sensor.is_valid_measurement(test_uvd), test_expected)

    def test_project_triangulate(self, sensor):
        test_xyz = [1., 2., 10.]
        test_uvd = [110., 120., 10.]
        assert np.allclose(sensor.triangulate(
            sensor.project(test_xyz)), test_xyz)
        assert np.allclose(sensor.project(
            sensor.triangulate(test_uvd)), test_uvd)

    def test_project_jacobian(self, sensor):
        test_xyz = [1., 2., 10.]
        expected_jacobian = np.array([[20., 0., -2.],
                                      [0., 20., -4.],
                                      [0., 0., -2.]])
        uvd, jacobian = sensor.project(test_xyz, compute_jacobians=True)
        assert np.allclose(jacobian, expected_jacobian)

    def test_triangulate_jacobian(self, sensor):
        test_uvd = [110., 120., 10.]
        expected_jacobian = np.array([[0.1, 0., -0.1],
                                      [0., 0.1, -0.2],
                                      [0., 0., -2.]])
        xyz, jacobian = sensor.triangulate(test_uvd, compute_jacobians=True)
        assert np.allclose(jacobian, expected_jacobian)

    def test_project_vectorized(self, sensor):
        test_xyz1 = [1., 2., 10.]
        test_xyz2 = [2., 1., -20.]
        test_xyz12 = np.array([test_xyz1, test_xyz2])  # 2x3
        uvd, jacobians = sensor.project(test_xyz12, True)
        assert uvd.shape == (2, 3)
        assert jacobians.shape == (2, 3, 3)
        assert np.all(np.isnan(uvd[1, :]))
        assert np.all(np.isnan(jacobians[1, :, :]))

    def test_triangulate_vectorized(self, sensor):
        test_uvd1 = [110., 120., 10.]
        test_uvd2 = [500., 600., 1.]
        test_uvd12 = np.array([test_uvd1, test_uvd2])  # 2x3
        xyz, jacobians = sensor.triangulate(test_uvd12, True)
        assert xyz.shape == (2, 3)
        assert jacobians.shape == (2, 3, 3)
        assert np.all(np.isnan(xyz[1, :]))
        assert np.all(np.isnan(jacobians[1, :, :]))
