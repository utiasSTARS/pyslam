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

    def test_project_triangulate(self, sensor):
        test_xyz = [1., 2., 10.]
        test_uvd = [110., 120., 10.]
        assert(
            np.allclose(sensor.triangulate(sensor.project(test_xyz)), test_xyz)
            and np.allclose(sensor.project(
                sensor.triangulate(test_uvd)), test_uvd)
        )

    def test_project_jacobian(self, sensor):
        test_xyz = [1., 2., 10.]
        expected_jacobian = np.array([[20., 0., -2.],
                                      [0., 20., -4.],
                                      [0., 0., -2.]])
        uvd, jacobian = sensor.project(test_xyz, compute_jacobians=True)
        assert(
            np.allclose(jacobian, expected_jacobian)
        )

    def test_triangulate_jacobian(self, sensor):
        test_uvd = [110., 120., 10.]
        expected_jacobian = np.array([[0.1, 0., -0.1],
                                      [0., 0.1, -0.2],
                                      [0., 0., -2.]])
        xyz, jacobian = sensor.triangulate(test_uvd, compute_jacobians=True)
        assert(
            np.allclose(jacobian, expected_jacobian)
        )

    def test_multi_triangulate(self, sensor):
        test_xyz1 = [1., 2., 10.]
        test_xyz2 = [2., 1., 20.]
        test_xyz12 = np.array([test_xyz1, test_xyz2])  # 2x3
        uvd, jacobians = sensor.triangulate(test_xyz12, True)
        assert(uvd.shape == (2, 3) and jacobians.shape == (2, 3, 3))
        uvd, jacobians = sensor.triangulate(test_xyz12, True)
        assert(uvd.shape == (2, 3) and jacobians.shape == (2, 3, 3))

    def test_multi_project(self, sensor):
        test_uvd1 = [110., 120., 10.]
        test_uvd2 = [500., 600., 1.]
        test_uvd12 = np.array([test_uvd1, test_uvd2])  # 2x3
        xyz, jacobians = sensor.triangulate(test_uvd12, True)
        assert(xyz.shape == (2, 3) and jacobians.shape == (2, 3, 3))
        xyz, jacobians = sensor.triangulate(test_uvd12, True)
        assert(xyz.shape == (2, 3) and jacobians.shape == (2, 3, 3))
