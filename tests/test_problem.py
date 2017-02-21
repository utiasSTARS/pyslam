import numpy as np
import pytest

from pyslam.problem import Options, Problem
from pyslam.utils import invsqrt


class TestBasic:

    def test_residual_blocks(self):
        from pyslam.costs import QuadraticCost
        problem = Problem()
        param_keys = ['a', 'b', 'c']
        problem.add_residual_block(QuadraticCost(2., 4., 1.),
                                   param_keys)
        assert param_keys == problem.block_param_keys[0]

    def test_param_dict(self):
        problem = Problem()
        params = {'a': 1, 'b': 2, 'c': 3}
        problem.initialize_params(params)
        assert(
            problem.param_dict == params
        )
        extra_param = {'d': 4}
        params.update(extra_param)
        problem.initialize_params(extra_param)
        assert problem.param_dict == params

    def test_constant_params(self):
        problem = Problem()
        problem.set_parameters_constant('a')
        assert problem.constant_param_keys == ['a']
        problem.set_parameters_constant(['a', 'b_param'])
        assert problem.constant_param_keys == ['a', 'b_param']
        problem.set_parameters_variable('a')
        assert problem.constant_param_keys == ['b_param']
        problem.set_parameters_variable('c')
        assert problem.constant_param_keys == ['b_param']
        problem.set_parameters_variable(['a', 'b_param', 'c'])
        assert problem.constant_param_keys == []

    def test_eval_cost(self):
        from pyslam.costs import QuadraticCost
        problem = Problem()
        good_params = {'a': 1., 'b': 2., 'c': 1.}
        bad_params = {'a': 1., 'b': 0., 'c': 0.}
        cost1 = QuadraticCost(1., 4., 0.5)
        cost2 = QuadraticCost(0., 1., 2.)
        problem.add_residual_block(cost1, ['a', 'b', 'c'])
        problem.add_residual_block(cost2, ['a', 'b', 'c'])
        problem.initialize_params(good_params)
        assert problem.eval_cost() == 0.
        assert problem.eval_cost(bad_params) == 0.5 * ((0.5 * 0.5 * 3. * 3.)
                                                       + (2. * 2. * 1. * 1.))

    def test_fit_quadratic(self):
        from pyslam.costs import QuadraticCost

        params_true = {'a': 1., 'b': -2., 'c': 3.}
        params_init = {'a': -20., 'b': 10., 'c': -30.}

        x_data = np.linspace(-5, 5, 10)
        y_data = params_true['a'] * x_data * x_data \
            + params_true['b'] * x_data + params_true['c']

        problem = Problem()
        for x, y in zip(x_data, y_data):
            problem.add_residual_block(QuadraticCost(
                x, y, 1.), ['a', 'b', 'c'])

        problem.initialize_params(params_init)
        params_final = problem.solve()

        for key in params_true.keys():
            assert(np.allclose(params_final[key], params_true[key]))


class TestPoseGraphRelax:

    @pytest.fixture
    def poses_true(self):
        from liegroups import SE2, SO2
        T_k_w = {
            'T_0_w': SE2.identity(),
            'T_1_w': SE2(SO2.identity(), -np.array([0.5, 0])),
            'T_2_w': SE2(SO2.identity(), -np.array([1, 0])),
            'T_3_w': SE2(SO2.from_angle(np.pi / 2),
                         -(SO2.from_angle(np.pi / 2) * np.array([1, 0.5]))),
            'T_4_w': SE2(SO2.from_angle(np.pi), -
                         (SO2.from_angle(np.pi) * np.array([0.5, 0.5]))),
            'T_5_w': SE2(SO2.from_angle(-np.pi / 2),
                         -(SO2.from_angle(-np.pi / 2) * np.array([0.5, 0])))
        }
        return T_k_w

    @pytest.fixture
    def poses_init(self, poses_true):
        from liegroups import SE2
        offset1 = SE2.exp([-0.1, 0.1, -0.1])
        offset2 = SE2.exp([0.1, -0.1, 0.1])
        T_k_w = {
            'T_0_w': poses_true['T_0_w'],
            'T_1_w': offset1 * poses_true['T_1_w'],
            'T_2_w': offset2 * poses_true['T_2_w'],
            'T_3_w': offset1 * poses_true['T_3_w'],
            'T_4_w': offset2 * poses_true['T_4_w'],
            'T_5_w': offset1 * poses_true['T_5_w'],
        }
        return T_k_w

    @pytest.fixture
    def odometry(self, poses_true):
        T_obs = {
            'T_0_w': poses_true['T_0_w'],
            'T_1_0': poses_true['T_1_w'] * poses_true['T_0_w'].inv(),
            'T_2_1': poses_true['T_2_w'] * poses_true['T_1_w'].inv(),
            'T_3_2': poses_true['T_3_w'] * poses_true['T_2_w'].inv(),
            'T_4_3': poses_true['T_4_w'] * poses_true['T_3_w'].inv(),
            'T_5_4': poses_true['T_5_w'] * poses_true['T_4_w'].inv(),
            'T_5_1': poses_true['T_5_w'] * poses_true['T_1_w'].inv()
        }
        return T_obs

    @pytest.fixture
    def costs(self, odometry):
        from pyslam.costs import PoseCost, PoseToPoseCost
        prior_stiffness = invsqrt(1e-12 * np.identity(3))
        odom_stiffness = invsqrt(1e-3 * np.identity(3))
        loop_stiffness = invsqrt(1. * np.identity(3))
        return [
            PoseCost(odometry['T_0_w'], prior_stiffness),
            PoseToPoseCost(odometry['T_1_0'], odom_stiffness),
            PoseToPoseCost(odometry['T_2_1'], odom_stiffness),
            PoseToPoseCost(odometry['T_3_2'], odom_stiffness),
            PoseToPoseCost(odometry['T_4_3'], odom_stiffness),
            PoseToPoseCost(odometry['T_5_4'], odom_stiffness),
            PoseToPoseCost(odometry['T_5_1'], loop_stiffness)
        ]

    @pytest.fixture
    def cost_params(self):
        return [
            ['T_0_w'],
            ['T_0_w', 'T_1_w'],
            ['T_1_w', 'T_2_w'],
            ['T_2_w', 'T_3_w'],
            ['T_3_w', 'T_4_w'],
            ['T_4_w', 'T_5_w'],
            ['T_1_w', 'T_5_w']
        ]

    @pytest.fixture
    def options(self):
        options = Options()
        options.allow_nondecreasing_steps = True
        options.max_nondecreasing_steps = 3
        return options

    def test_first_pose_constant(self, options, costs, cost_params,
                                 poses_init, poses_true):
        from liegroups import SE2
        problem = Problem(options)
        for i in range(1, 6):
            problem.add_residual_block(costs[i], cost_params[i])
        problem.set_parameters_constant('T_0_w')
        problem.initialize_params(poses_init)
        poses_final = problem.solve()
        for key in poses_true.keys():
            assert np.linalg.norm(
                SE2.log(poses_final[key].inv() * poses_true[key])) < 1e-4

    def test_first_pose_prior(self, options, costs, cost_params,
                              poses_init, poses_true):
        from liegroups import SE2
        problem = Problem(options)
        for i in range(0, 6):
            problem.add_residual_block(costs[i], cost_params[i])
        problem.initialize_params(poses_init)
        poses_final = problem.solve()
        for key in poses_true.keys():
            assert np.linalg.norm(
                SE2.log(poses_final[key].inv() * poses_true[key])) < 1e-4

    def test_loop_closure(self, options, costs, cost_params,
                          poses_init, poses_true):
        from liegroups import SE2
        problem = Problem(options)
        for i in range(0, 7):
            problem.add_residual_block(costs[i], cost_params[i])
        problem.initialize_params(poses_init)
        poses_final = problem.solve()
        for key in poses_true.keys():
            assert np.linalg.norm(
                SE2.log(poses_final[key].inv() * poses_true[key])) < 1e-4


class TestBundleAdjust:

    @pytest.fixture
    def options(self):
        options = Options()
        options.allow_nondecreasing_steps = True
        options.max_nondecreasing_steps = 3
        return options

    @pytest.fixture
    def camera(self):
        from pyslam.sensors import StereoCamera
        return StereoCamera(640, 480, 1000, 1000, 0.25, 1280, 960)

    @pytest.fixture
    def points(self):
        pts_w_true = [
            np.array([0., -1., 10.]),
            np.array([1., 1., 5.]),
            np.array([-1., 1., 15.])
        ]
        return pts_w_true

    @pytest.fixture
    def poses(self):
        from liegroups import SE3
        T_cam_w_true = [
            SE3.identity(),
            SE3.exp(0.1 * np.ones(6)),
            SE3.exp(0.2 * np.ones(6)),
            SE3.exp(0.3 * np.ones(6))
        ]
        return T_cam_w_true

    @pytest.fixture
    def observations(self, camera, points, poses):
        return [[camera.project(T * p) for p in points] for T in poses]

    def test_bundle_adjust(self, options, camera, points, poses, observations):
        from liegroups import SE3
        from pyslam.costs import ReprojectionCost

        problem = Problem(options)

        obs_var = [1, 1, 2]  # [u,v,d]
        obs_stiffness = invsqrt(np.diagflat(obs_var))

        for i, this_pose_obs in enumerate(observations):
            for j, o in enumerate(this_pose_obs):
                cost = ReprojectionCost(camera, o, obs_stiffness)
                problem.add_residual_block(
                    cost, ['T_cam{}_w'.format(i), 'pt{}_w'.format(j)])

        params_true = {}
        params_init = {}

        for i, pt in enumerate(points):
            pid = 'pt{}_w'.format(i)
            params_true.update({pid: pt})
            params_init.update({pid: camera.triangulate(
                observations[0][i] + 10. * np.random.rand(3))})

        for i, pose in enumerate(poses):
            pid = 'T_cam{}_w'.format(i)
            params_true.update({pid: pose})
            params_init.update({pid: SE3.identity()})

        problem.initialize_params(params_init)
        problem.set_parameters_constant('T_cam0_w')

        params_final = problem.solve()

        for key in params_true.keys():
            p_est = params_final[key]
            p_true = params_true[key]

            if isinstance(p_est, SE3):
                err = SE3.log(p_est.inv() * p_true)
            else:
                err = p_est - p_true

            assert np.linalg.norm(err) < 1e-4


class TestCovariance:

    @pytest.fixture
    def options(self):
        options = Options()
        options.allow_nondecreasing_steps = True
        options.max_nondecreasing_steps = 3
        return options

    def test_se3(self, options):
        from liegroups import SE3
        from pyslam.costs import PoseCost, PoseToPoseCost

        problem = Problem(options)

        odom = SE3.exp(0.1 * np.ones(6))

        odom_stiffness = invsqrt(1e-3 * np.eye(SE3.dof))
        T0_stiffness = invsqrt(1e-6 * np.eye(SE3.dof))

        odom_covar = np.linalg.inv(np.dot(odom_stiffness, odom_stiffness))
        T0_covar = np.linalg.inv(np.dot(T0_stiffness, T0_stiffness))

        cost0 = PoseCost(SE3.identity(), T0_stiffness)
        cost1 = PoseToPoseCost(odom, odom_stiffness)

        params_init = {'T0': SE3.identity(), 'T1': SE3.identity()}

        problem.add_residual_block(cost0, 'T0')
        problem.add_residual_block(cost1, ['T0', 'T1'])
        problem.initialize_params(params_init)
        problem.solve()
        problem.compute_covariance()
        estimated_covar = problem.get_covariance_block('T1', 'T1')
        expected_covar = odom_covar + odom.adjoint().dot(T0_covar.dot(odom.adjoint().T))

        assert np.allclose(estimated_covar, expected_covar)
