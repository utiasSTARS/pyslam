import pytest

import numpy as np

from liegroups import SE2, SE3


class TestQuadraticCost:

    @pytest.fixture
    def cost(self):
        from pyslam.costs import QuadraticCost
        return QuadraticCost(2., 3., 1.)

    def test_evaluate(self, cost):
        params_good = [1., -2., 3.]
        params_bad = [0., 3., 1.]
        assert cost.evaluate(params_good) == 0.
        assert cost.evaluate(params_bad) != 0.

    def test_jacobians(self, cost):
        params = [1., -2., 3.]
        expected_jac = [4., 2., 1.]
        _, jac1 = cost.evaluate(params, compute_jacobians=[True, True, True])
        _, jac2 = cost.evaluate(params, compute_jacobians=[
                                False, False, False])
        assert len(jac1) == len(jac2) == 3
        assert np.allclose(jac1, expected_jac)
        assert not any(jac2)


class TestPoseCost:

    @pytest.fixture
    def se2_cost(self):
        from pyslam.costs import PoseCost
        return PoseCost(SE2.exp(np.array([1, 2, 3])), np.eye(3))

    def test_evaluate_se2(self, se2_cost):
        T_same = se2_cost.T_obs
        T_diff = SE2.exp([4, 5, 6])
        assert np.allclose(se2_cost.evaluate([T_same]), np.zeros(3))
        assert not np.allclose(se2_cost.evaluate([T_diff]), np.zeros(3))

    def test_jacobians_se2(self, se2_cost):
        T_test = SE2.exp([4, 5, 6])
        _, jacobians = se2_cost.evaluate(
            [T_test], compute_jacobians=[True])
        assert len(jacobians) == 1 and jacobians[0].shape == (3, 3)

    @pytest.fixture
    def se3_cost(self):
        from pyslam.costs import PoseCost
        return PoseCost(SE3.exp([1, 2, 3, 4, 5, 6]), np.eye(6))

    def test_evaluate_se3(self, se3_cost):
        T_same = se3_cost.T_obs
        T_diff = SE3.exp([7, 8, 9, 10, 11, 12])
        assert np.allclose(se3_cost.evaluate([T_same]), np.zeros(6))
        assert not np.allclose(se3_cost.evaluate([T_diff]), np.zeros(6))

    def test_jacobians_se3(self, se3_cost):
        T_test = SE3.exp([7, 8, 9, 10, 11, 12])
        _, jacobians = se3_cost.evaluate(
            [T_test], compute_jacobians=[True])
        assert len(jacobians) == 1 and jacobians[0].shape == (6, 6)


class TestPoseToPoseCost:

    @pytest.fixture
    def se2_cost(self):
        from pyslam.costs import PoseToPoseCost
        return PoseToPoseCost(SE2.identity(), np.eye(3))

    def test_evaluate_se2(self, se2_cost):
        T1 = SE2.exp([1, 2, 3])
        T2 = SE2.exp([4, 5, 6])
        assert np.allclose(se2_cost.evaluate([T1, T1]), np.zeros(3))
        assert not np.allclose(se2_cost.evaluate([T1, T2]), np.zeros(3))

    def test_jacobians_se2(self, se2_cost):
        T1 = SE2.exp([1, 2, 3])
        T2 = SE2.exp([4, 5, 6])
        _, jac1 = se2_cost.evaluate(
            [T1, T2], compute_jacobians=[True, True])
        _, jac2 = se2_cost.evaluate(
            [T1, T2], compute_jacobians=[True, False])
        _, jac3 = se2_cost.evaluate(
            [T1, T2], compute_jacobians=[False, True])
        assert len(jac1) == len(jac2) == len(jac3) == 2
        assert jac1[0].shape == (3, 3) and jac1[1].shape == (3, 3)
        assert jac2[0].shape == (3, 3) and jac2[1] is None
        assert jac3[0] is None and jac3[1].shape == (3, 3)

    @pytest.fixture
    def se3_cost(self):
        from pyslam.costs import PoseToPoseCost
        return PoseToPoseCost(SE3.identity(), np.eye(6))

    def test_evaluate_se3(self, se3_cost):
        T1 = SE3.exp([1, 2, 3, 4, 5, 6])
        T2 = SE3.exp([7, 8, 9, 10, 11, 12])
        assert np.allclose(se3_cost.evaluate([T1, T1]), np.zeros(6))
        assert not np.allclose(se3_cost.evaluate([T1, T2]), np.zeros(6))

    def test_jacobians_se3(self, se3_cost):
        T1 = SE3.exp([1, 2, 3, 4, 5, 6])
        T2 = SE3.exp([7, 8, 9, 10, 11, 12])
        _, jac1 = se3_cost.evaluate(
            [T1, T2], compute_jacobians=[True, True])
        _, jac2 = se3_cost.evaluate(
            [T1, T2], compute_jacobians=[True, False])
        _, jac3 = se3_cost.evaluate(
            [T1, T2], compute_jacobians=[False, True])
        assert len(jac1) == len(jac2) == len(jac3) == 2
        assert jac1[0].shape == (6, 6) and jac1[1].shape == (6, 6)
        assert jac2[0].shape == (6, 6) and jac2[1] is None
        assert jac3[0] is None and jac3[1].shape == (6, 6)


class TestReprojectionCost:

    @pytest.fixture
    def stereo_cost(self):
        from pyslam.sensors import StereoCamera
        from pyslam.costs import ReprojectionCost
        cu = 100.
        cv = 100.
        fu = 200.
        fv = 200.
        b = 1.
        w = 200
        h = 200
        return ReprojectionCost(StereoCamera(cu, cv, fu, fv, b, w, h),
                                [40, 60, 10], np.eye(3))

    def test_evaluate_stereo(self, stereo_cost):
        T_cam_w = SE3.exp([1, 2, 3, 4, 5, 6])
        pt_good_w = T_cam_w.inv() \
            * stereo_cost.camera.triangulate(stereo_cost.obs)
        pt_bad_w = pt_good_w + [1, 1, 1]
        assert np.allclose(stereo_cost.evaluate(
            [T_cam_w, pt_good_w]), np.zeros(3))
        assert not np.allclose(stereo_cost.evaluate(
            [T_cam_w, pt_bad_w]), np.zeros(3))

    def test_jacobians_stereo(self, stereo_cost):
        T_cam_w = SE3.exp([1, 2, 3, 4, 5, 6])
        pt_w = T_cam_w.inv() * stereo_cost.camera.triangulate(stereo_cost.obs)
        _, jac1 = stereo_cost.evaluate(
            [T_cam_w, pt_w], compute_jacobians=[True, True])
        _, jac2 = stereo_cost.evaluate(
            [T_cam_w, pt_w], compute_jacobians=[True, False])
        _, jac3 = stereo_cost.evaluate(
            [T_cam_w, pt_w], compute_jacobians=[False, True])
        assert len(jac1) == len(jac2) == len(jac3) == 2
        assert jac1[0].shape == (3, 6) and jac1[1].shape == (3, 3)
        assert jac2[0].shape == (3, 6) and jac2[1] is None
        assert jac3[0] is None and jac3[1].shape == (3, 3)
