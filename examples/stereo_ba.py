import numpy as np

from liegroups import SE3

from pyslam.sensors import StereoCamera
from pyslam.costs import ReprojectionCost
from pyslam.problem import Options, Problem

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
obs_covar = np.diagflat(np.array(obs_var))

obs = [[camera.project(T * p)
        for p in pts_w_GT] for T in T_cam_w_GT]

# Optimize
options = Options()
options.allow_nondecreasing_steps = True
options.max_nondecreasing_steps = 3

problem = Problem(options)

for i in range(len(obs)):
    for j in range(len(obs[i])):
        cost = ReprojectionCost(camera, obs[i][j], np.linalg.inv(obs_covar))
        problem.add_residual_block(
            cost, ['T_cam{}_w'.format(i), 'pt{}_w'.format(j)])

params_true = {}
params_init = {}

for i in range(len(pts_w_GT)):
    pid = 'pt{}_w'.format(i)
    params_true.update({pid: pts_w_GT[i]})
    params_init.update({pid: camera.triangulate(obs[0][i])})

for i in range(len(T_cam_w_GT)):
    pid = 'T_cam{}_w'.format(i)
    params_true.update({pid: T_cam_w_GT[i]})
    params_init.update({pid: SE3.identity()})

problem.initialize_params(params_init)
problem.set_parameters_constant('T_cam0_w')

params_final = problem.solve()
print()

# Compute errors
print("Initial Error:")
for key in params_true.keys():
    p_est = params_init[key]
    p_true = params_true[key]

    if isinstance(p_est, SE3):
        err = SE3.log(p_est.inv() * p_true)
    else:
        err = p_est - p_true

    print('{}: {}'.format(key, err))

print()

print("Final Error:")
for key in params_true.keys():
    p_est = params_final[key]
    p_true = params_true[key]

    if isinstance(p_est, SE3):
        err = SE3.log(p_est.inv() * p_true)
    else:
        err = p_est - p_true

    print('{}: {}'.format(key, err))
