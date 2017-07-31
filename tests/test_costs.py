import pytest

import numpy as np

from liegroups import SE2, SE3


class TestQuadraticResidual:

    @pytest.fixture
    def residual(self):
        from pyslam.residuals import QuadraticResidual
        return QuadraticResidual(2., 3., 1.)

    def test_evaluate(self, residual):
        params_good = [1., -2., 3.]
        params_bad = [0., 3., 1.]
        assert residual.evaluate(params_good) == 0.
        assert residual.evaluate(params_bad) != 0.

    def test_jacobians(self, residual):
        params = [1., -2., 3.]
        expected_jac = [4., 2., 1.]
        _, jac1 = residual.evaluate(
            params, compute_jacobians=[True, True, True])
        _, jac2 = residual.evaluate(params, compute_jacobians=[
            False, False, False])
        assert len(jac1) == len(jac2) == 3
        assert np.allclose(jac1, expected_jac)
        assert not any(jac2)


class TestPoseResidual:

    @pytest.fixture
    def se2_residual(self):
        from pyslam.residuals import PoseResidual
        return PoseResidual(SE2.exp(np.array([1, 2, 3])), np.eye(3))

    def test_evaluate_se2(self, se2_residual):
        T_same = se2_residual.T_obs
        T_diff = SE2.exp([4, 5, 6])
        assert np.allclose(se2_residual.evaluate([T_same]), np.zeros(3))
        assert not np.allclose(se2_residual.evaluate([T_diff]), np.zeros(3))

    def test_jacobians_se2(self, se2_residual):
        T_test = SE2.exp([4, 5, 6])
        _, jacobians = se2_residual.evaluate(
            [T_test], compute_jacobians=[True])
        assert len(jacobians) == 1 and jacobians[0].shape == (3, 3)

    @pytest.fixture
    def se3_residual(self):
        from pyslam.residuals import PoseResidual
        return PoseResidual(SE3.exp([1, 2, 3, 4, 5, 6]), np.eye(6))

    def test_evaluate_se3(self, se3_residual):
        T_same = se3_residual.T_obs
        T_diff = SE3.exp([7, 8, 9, 10, 11, 12])
        assert np.allclose(se3_residual.evaluate([T_same]), np.zeros(6))
        assert not np.allclose(se3_residual.evaluate([T_diff]), np.zeros(6))

    def test_jacobians_se3(self, se3_residual):
        T_test = SE3.exp([7, 8, 9, 10, 11, 12])
        _, jacobians = se3_residual.evaluate(
            [T_test], compute_jacobians=[True])
        assert len(jacobians) == 1 and jacobians[0].shape == (6, 6)


class TestPoseToPoseResidual:

    @pytest.fixture
    def se2_residual(self):
        from pyslam.residuals import PoseToPoseResidual
        return PoseToPoseResidual(SE2.identity(), np.eye(3))

    def test_evaluate_se2(self, se2_residual):
        T1 = SE2.exp([1, 2, 3])
        T2 = SE2.exp([4, 5, 6])
        assert np.allclose(se2_residual.evaluate([T1, T1]), np.zeros(3))
        assert not np.allclose(se2_residual.evaluate([T1, T2]), np.zeros(3))

    def test_jacobians_se2(self, se2_residual):
        T1 = SE2.exp([1, 2, 3])
        T2 = SE2.exp([4, 5, 6])
        _, jac1 = se2_residual.evaluate(
            [T1, T2], compute_jacobians=[True, True])
        _, jac2 = se2_residual.evaluate(
            [T1, T2], compute_jacobians=[True, False])
        _, jac3 = se2_residual.evaluate(
            [T1, T2], compute_jacobians=[False, True])
        assert len(jac1) == len(jac2) == len(jac3) == 2
        assert jac1[0].shape == (3, 3) and jac1[1].shape == (3, 3)
        assert jac2[0].shape == (3, 3) and jac2[1] is None
        assert jac3[0] is None and jac3[1].shape == (3, 3)

    @pytest.fixture
    def se3_residual(self):
        from pyslam.residuals import PoseToPoseResidual
        return PoseToPoseResidual(SE3.identity(), np.eye(6))

    def test_evaluate_se3(self, se3_residual):
        T1 = SE3.exp([1, 2, 3, 4, 5, 6])
        T2 = SE3.exp([7, 8, 9, 10, 11, 12])
        assert np.allclose(se3_residual.evaluate([T1, T1]), np.zeros(6))
        assert not np.allclose(se3_residual.evaluate([T1, T2]), np.zeros(6))

    def test_jacobians_se3(self, se3_residual):
        T1 = SE3.exp([1, 2, 3, 4, 5, 6])
        T2 = SE3.exp([7, 8, 9, 10, 11, 12])
        _, jac1 = se3_residual.evaluate(
            [T1, T2], compute_jacobians=[True, True])
        _, jac2 = se3_residual.evaluate(
            [T1, T2], compute_jacobians=[True, False])
        _, jac3 = se3_residual.evaluate(
            [T1, T2], compute_jacobians=[False, True])
        assert len(jac1) == len(jac2) == len(jac3) == 2
        assert jac1[0].shape == (6, 6) and jac1[1].shape == (6, 6)
        assert jac2[0].shape == (6, 6) and jac2[1] is None
        assert jac3[0] is None and jac3[1].shape == (6, 6)


class TestReprojectionResidual:

    @pytest.fixture
    def stereo_residual(self):
        from pyslam.sensors import StereoCamera
        from pyslam.residuals import ReprojectionResidual
        cu = 100.
        cv = 100.
        fu = 200.
        fv = 200.
        b = 1.
        w = 200
        h = 200
        return ReprojectionResidual(StereoCamera(cu, cv, fu, fv, b, w, h),
                                    [40, 60, 10], np.eye(3))

    def test_evaluate_stereo(self, stereo_residual):
        T_cam_w = SE3.exp([1, 2, 3, 4, 5, 6])
        pt_good_w = T_cam_w.inv().dot(
            stereo_residual.camera.triangulate(stereo_residual.obs))
        pt_bad_w = pt_good_w + [1, 1, 1]
        assert np.allclose(stereo_residual.evaluate(
            [T_cam_w, pt_good_w]), np.zeros(3))
        assert not np.allclose(stereo_residual.evaluate(
            [T_cam_w, pt_bad_w]), np.zeros(3))

    def test_jacobians_stereo(self, stereo_residual):
        T_cam_w = SE3.exp([1, 2, 3, 4, 5, 6])
        pt_w = T_cam_w.inv().dot(
            stereo_residual.camera.triangulate(stereo_residual.obs))
        _, jac1 = stereo_residual.evaluate(
            [T_cam_w, pt_w], compute_jacobians=[True, True])
        _, jac2 = stereo_residual.evaluate(
            [T_cam_w, pt_w], compute_jacobians=[True, False])
        _, jac3 = stereo_residual.evaluate(
            [T_cam_w, pt_w], compute_jacobians=[False, True])
        assert len(jac1) == len(jac2) == len(jac3) == 2
        assert jac1[0].shape == (3, 6) and jac1[1].shape == (3, 3)
        assert jac2[0].shape == (3, 6) and jac2[1] is None
        assert jac3[0] is None and jac3[1].shape == (3, 3)
