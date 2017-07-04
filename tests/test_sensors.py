import pytest
import numpy as np


def get_pinhole_intrinsics():
    cu = 150.
    cv = 100.
    fu = 250.
    fv = 200.
    w = 300
    h = 200
    return cu, cv, fu, fv, w, h


class TestStereoCamera:

    @pytest.fixture
    def sensor(self):
        from pyslam.sensors import StereoCamera
        cu, cv, fu, fv, w, h = get_pinhole_intrinsics()
        b = 1.

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
        expected_jacobian = np.array([[25., 0., -2.5],
                                      [0., 20., -4.],
                                      [0., 0., -2.5]])
        uvd, jacobian = sensor.project(test_xyz, compute_jacobians=True)
        assert np.allclose(jacobian, expected_jacobian)

    def test_triangulate_jacobian(self, sensor):
        test_uvd = [110., 120., 10.]
        expected_jacobian = np.array([[0.1, 0., 0.4],
                                      [0., 0.125, -0.25],
                                      [0., 0., -2.5]])
        xyz, jacobian = sensor.triangulate(test_uvd, compute_jacobians=True)
        assert np.allclose(jacobian, expected_jacobian)

    def test_project_vectorized(self, sensor):
        test_xyz1 = [1., 2., 10.]
        test_xyz2 = [2., 1., -20.]
        test_xyz12 = np.array([test_xyz1, test_xyz2])  # 2x3
        uvd, jacobians = sensor.project(test_xyz12, True)
        assert uvd.shape == (2, 3)
        assert jacobians.shape == (2, 3, 3)

    def test_triangulate_vectorized(self, sensor):
        test_uvd1 = [110., 120., 10.]
        test_uvd2 = [500., 600., 1.]
        test_uvd12 = np.array([test_uvd1, test_uvd2])  # 2x3
        xyz, jacobians = sensor.triangulate(test_uvd12, True)
        assert xyz.shape == (2, 3)
        assert jacobians.shape == (2, 3, 3)


class TestRGBDCamera:

    @pytest.fixture
    def sensor(self):
        from pyslam.sensors import RGBDCamera
        cu, cv, fu, fv, w, h = get_pinhole_intrinsics()
        return RGBDCamera(cu, cv, fu, fv, w, h)

    def test_is_valid_measurement(self, sensor):
        test_uvz = np.array([[110., 120., 10.],
                             [-10., 100., 10.],
                             [0., -10., 10.],
                             [0., 0., -5.]])
        test_expected = [True, False, False, False]
        for uvz, expected in zip(test_uvz, test_expected):
            assert sensor.is_valid_measurement(uvz) == expected

        assert np.array_equal(
            sensor.is_valid_measurement(test_uvz), test_expected)

    def test_project_triangulate(self, sensor):
        test_xyz = [1., 2., 10.]
        test_uvz = [110., 120., 10.]
        assert np.allclose(sensor.triangulate(
            sensor.project(test_xyz)), test_xyz)
        assert np.allclose(sensor.project(
            sensor.triangulate(test_uvz)), test_uvz)

    def test_project_jacobian(self, sensor):
        test_xyz = [1., 2., 10.]
        expected_jacobian = np.array([[25., 0., -2.5],
                                      [0., 20., -4.],
                                      [0., 0., 1.]])
        uvz, jacobian = sensor.project(test_xyz, compute_jacobians=True)
        assert np.allclose(jacobian, expected_jacobian)

    def test_triangulate_jacobian(self, sensor):
        test_uvz = [110., 120., 10.]
        expected_jacobian = np.array([[0.04, 0., -0.16],
                                      [0., 0.05, 0.1],
                                      [0., 0., 1.]])
        xyz, jacobian = sensor.triangulate(test_uvz, compute_jacobians=True)
        assert np.allclose(jacobian, expected_jacobian)

    def test_project_vectorized(self, sensor):
        test_xyz1 = [1., 2., 10.]
        test_xyz2 = [2., 1., -20.]
        test_xyz12 = np.array([test_xyz1, test_xyz2])  # 2x3
        uvz, jacobians = sensor.project(test_xyz12, True)
        assert uvz.shape == (2, 3)
        assert jacobians.shape == (2, 3, 3)

    def test_triangulate_vectorized(self, sensor):
        test_uvz1 = [110., 120., 10.]
        test_uvz2 = [500., 600., 1.]
        test_uvz12 = np.array([test_uvz1, test_uvz2])  # 2x3
        xyz, jacobians = sensor.triangulate(test_uvz12, True)
        assert xyz.shape == (2, 3)
        assert jacobians.shape == (2, 3, 3)
