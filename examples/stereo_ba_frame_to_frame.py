import numpy as np

from liegroups import SE3

from pyslam.sensors import StereoCamera
from pyslam.residuals import ReprojectionResidualFrameToFrame
from pyslam.problem import Options, Problem
from pyslam.utils import invsqrt
from collections import OrderedDict

# This example performs frame-to-frame bundle adjustment, optimizing only
# for the relative poses (and not the 3D landmarks)

# Reproducibility
np.random.seed(42)

# Landmarks (world frame is first camera frame)
pts_w_GT = [
    np.array([0., -1., 10.]),
    np.array([1., 1., 5.]),
    np.array([-1., 1., 15.])
]

# Trajectory
T_cam_w_GT = [
    SE3.identity(),
    SE3.exp(0.1 * np.ones(6)),
    SE3.exp(0.2 * np.ones(6)),
    SE3.exp(0.3 * np.ones(6))
]

# Camera
camera = StereoCamera(640, 480, 1000, 1000, 0.25, 1280, 960)

# Observations
obs_var = [1, 1, 2]  # [u,v,d]
obs_stiffness = invsqrt(np.diagflat(obs_var))

# Collect all observations of the landmarks
obs = [[camera.project(T * p)
        for p in pts_w_GT] for T in T_cam_w_GT]

# Options
options = Options()
options.allow_nondecreasing_steps = True
options.max_nondecreasing_steps = 3

problem = Problem(options)

# Collect observations in pairs of poses (i.e. 0-1, 1-2, 2-3) and add
# residuals to problem
for i in range(len(obs) - 1):
    pose_1_obs = obs[i]
    pose_2_obs = obs[i + 1]
    for j, o_1 in enumerate(pose_1_obs):
        o_2 = pose_2_obs[j]
        residual = ReprojectionResidualFrameToFrame(
            camera, o_1, o_2, obs_stiffness)
        problem.add_residual_block(
            residual, ['T_cam{}_cam{}'.format(i + 1, i)])

params_true = OrderedDict({})
params_init = OrderedDict({})

# Initialize the relative SE(3) transforms
for i in range(len(obs) - 1):
    T_c1_w = T_cam_w_GT[i]
    T_c2_w = T_cam_w_GT[i + 1]
    pid = 'T_cam{}_cam{}'.format(i + 1, i)
    params_true.update({pid: T_c2_w * T_c1_w.inv()})
    params_init.update({pid: SE3.identity()})

problem.initialize_params(params_init)
params_final = problem.solve()

print()

# Compute errors
print("Initial Error:")
for key in params_true.keys():
    p_est = params_init[key]
    p_true = params_true[key]

    err = SE3.log(p_est.inv() * p_true)
    print('{}: {}'.format(key, err))

print()

print("Final Error:")
for key in params_true.keys():
    p_est = params_final[key]
    p_true = params_true[key]

    err = SE3.log(p_est.inv() * p_true)
    print('{}: {}'.format(key, err))
