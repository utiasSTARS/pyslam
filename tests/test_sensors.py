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

    def test_measurement_validation(self, sensor):
        uvd_good = [0., 0., 1.]
        uvd_bad1 = [-1., 0., 1.]
        uvd_bad2 = [0., -1., 1.]
        uvd_bad3 = [201., 0., 1.]
        uvd_bad4 = [0., 201., 1.]
        uvd_bad5 = [0., 0., 0.]
        assert(
            sensor.is_valid_measurement(uvd_good)
            and not sensor.is_valid_measurement(uvd_bad1)
            and not sensor.is_valid_measurement(uvd_bad2)
            and not sensor.is_valid_measurement(uvd_bad3)
            and not sensor.is_valid_measurement(uvd_bad4)
            and not sensor.is_valid_measurement(uvd_bad5)
        )

    def test_invalid_project(self, sensor):
        bad_xyz = np.array([0., 0., -1.])
        with pytest.raises(Exception):
            sensor.project(bad_xyz)

    def test_invalid_triangulate(self, sensor):
        bad_uvd = np.array([0., 0., -1.])
        with pytest.raises(Exception):
            sensor.triangulate(bad_uvd)

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
