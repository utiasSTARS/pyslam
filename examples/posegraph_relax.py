# import copy

import numpy as np

from liegroups import SE3, SO3

from pyslam.residuals import PoseResidual, PoseToPoseResidual
from pyslam.problem import Options, Problem
from pyslam.utils import invsqrt

T_1_0_true = SE3.identity()
T_2_0_true = SE3(SO3.identity(), -np.array([0.5, 0, 0]))
T_3_0_true = SE3(SO3.identity(), -np.array([1, 0, 0]))
T_4_0_true = SE3(SO3.rotz(np.pi / 2),
                 -(SO3.rotz(np.pi / 2).dot(np.array([1, 0.5, 0]))))
T_5_0_true = SE3(SO3.rotz(np.pi), -
                 (SO3.rotz(np.pi).dot(np.array([0.5, 0.5, 0]))))
T_6_0_true = SE3(SO3.rotz(-np.pi / 2),
                 -(SO3.rotz(-np.pi / 2).dot(np.array([0.5, 0, 0]))))
# T_6_0_true = copy.deepcopy(T_2_0_true)

# Odometry
T_1_0_obs = SE3.identity()
T_2_1_obs = T_2_0_true.dot(T_1_0_true.inv())
T_3_2_obs = T_3_0_true.dot(T_2_0_true.inv())
T_4_3_obs = T_4_0_true.dot(T_3_0_true.inv())
T_5_4_obs = T_5_0_true.dot(T_4_0_true.inv())
T_6_5_obs = T_6_0_true.dot(T_5_0_true.inv())

# Loop closure
T_6_2_obs = T_6_0_true.dot(T_2_0_true.inv())

# Random start
# T_1_0_init = SE3.exp(0.1 * 2 * (np.random.rand(6) - 0.5)) * T_1_0_true
# T_2_0_init = SE3.exp(0.1 * 2 * (np.random.rand(6) - 0.5)) * T_2_0_true
# T_3_0_init = SE3.exp(0.1 * 2 * (np.random.rand(6) - 0.5)) * T_3_0_true
# T_4_0_init = SE3.exp(0.1 * 2 * (np.random.rand(6) - 0.5)) * T_4_0_true
# T_5_0_init = SE3.exp(0.1 * 2 * (np.random.rand(6) - 0.5)) * T_5_0_true
# T_6_0_init = SE3.exp(0.1 * 2 * (np.random.rand(6) - 0.5)) * T_6_0_true

# Constant wrong start
offset1 = np.array([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1])
offset2 = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
T_1_0_init = SE3.exp(offset2).dot(T_1_0_true)
T_2_0_init = SE3.exp(offset1).dot(T_2_0_true)
T_3_0_init = SE3.exp(offset2).dot(T_3_0_true)
T_4_0_init = SE3.exp(offset1).dot(T_4_0_true)
T_5_0_init = SE3.exp(offset2).dot(T_5_0_true)
T_6_0_init = SE3.exp(offset1).dot(T_6_0_true)


# Either we need a prior on the first pose, or it needs to be held constant
# so that the resulting system of linear equations is solveable
prior_stiffness = invsqrt(1e-12 * np.identity(6))
odom_stiffness = invsqrt(1e-3 * np.identity(6))
loop_stiffness = invsqrt(1. * np.identity(6))

residual0 = PoseResidual(T_1_0_obs, prior_stiffness)
residual0_params = ['T_1_0']

residual1 = PoseToPoseResidual(T_2_1_obs, odom_stiffness)
residual1_params = ['T_1_0', 'T_2_0']

residual2 = PoseToPoseResidual(T_3_2_obs, odom_stiffness)
residual2_params = ['T_2_0', 'T_3_0']

residual3 = PoseToPoseResidual(T_4_3_obs, odom_stiffness)
residual3_params = ['T_3_0', 'T_4_0']

residual4 = PoseToPoseResidual(T_5_4_obs, odom_stiffness)
residual4_params = ['T_4_0', 'T_5_0']

residual5 = PoseToPoseResidual(T_6_5_obs, odom_stiffness)
residual5_params = ['T_5_0', 'T_6_0']

residual6 = PoseToPoseResidual(T_6_2_obs, loop_stiffness)
residual6_params = ['T_2_0', 'T_6_0']

options = Options()
options.allow_nondecreasing_steps = True
options.max_nondecreasing_steps = 3

problem = Problem(options)
problem.add_residual_block(residual0, residual0_params)
problem.add_residual_block(residual1, residual1_params)
problem.add_residual_block(residual2, residual2_params)
problem.add_residual_block(residual3, residual3_params)
problem.add_residual_block(residual4, residual4_params)
problem.add_residual_block(residual5, residual5_params)
problem.add_residual_block(residual6, residual6_params)

# problem.set_parameters_constant(residual0_params)
# problem.set_parameters_constant(residual1_params)

params_init = {'T_1_0': T_1_0_init, 'T_2_0': T_2_0_init,
               'T_3_0': T_3_0_init, 'T_4_0': T_4_0_init,
               'T_5_0': T_5_0_init, 'T_6_0': T_6_0_init}
params_true = {'T_1_0': T_1_0_true, 'T_2_0': T_2_0_true,
               'T_3_0': T_3_0_true, 'T_4_0': T_4_0_true,
               'T_5_0': T_5_0_true, 'T_6_0': T_6_0_true}

problem.initialize_params(params_init)

params_final = problem.solve()
print(problem.summary(format='full'))

print("Initial Error:")
for key in params_true.keys():
    print('{}: {}'.format(key, SE3.log(
        params_init[key].inv().dot(params_true[key]))))

print()

print("Final Error:")
for key in params_true.keys():
    print('{}: {}'.format(key, SE3.log(
        params_final[key].inv().dot(params_true[key]))))
