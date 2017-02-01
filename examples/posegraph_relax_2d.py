# import copy

import numpy as np

from liegroups import SE2, SO2

from pyslam.costs import PoseCost, PoseToPoseCost
from pyslam.problem import Options, Problem

T_1_0_true = SE2.identity()
T_2_0_true = SE2(SO2.identity(), -np.array([0.5, 0]))
T_3_0_true = SE2(SO2.identity(), -np.array([1, 0]))
T_4_0_true = SE2(SO2.fromangle(np.pi / 2),
                 -(SO2.fromangle(np.pi / 2) * np.array([1, 0.5])))
T_5_0_true = SE2(SO2.fromangle(np.pi), -
                 (SO2.fromangle(np.pi) * np.array([0.5, 0.5])))
T_6_0_true = SE2(SO2.fromangle(-np.pi / 2),
                 -(SO2.fromangle(-np.pi / 2) * np.array([0.5, 0])))
# T_6_0_true = copy.deepcopy(T_2_0_true)

# Odometry
T_1_0_obs = SE2.identity()
T_2_1_obs = T_2_0_true * T_1_0_true.inv()
T_3_2_obs = T_3_0_true * T_2_0_true.inv()
T_4_3_obs = T_4_0_true * T_3_0_true.inv()
T_5_4_obs = T_5_0_true * T_4_0_true.inv()
T_6_5_obs = T_6_0_true * T_5_0_true.inv()

# Loop closure
T_6_2_obs = T_6_0_true * T_2_0_true.inv()

# Random start
# T_1_0_init = SE2.exp(0.1 * 2 * (np.random.rand(3) - 0.5)) * T_1_0_true
# T_2_0_init = SE2.exp(0.1 * 2 * (np.random.rand(3) - 0.5)) * T_2_0_true
# T_3_0_init = SE2.exp(0.1 * 2 * (np.random.rand(3) - 0.5)) * T_3_0_true
# T_4_0_init = SE2.exp(0.1 * 2 * (np.random.rand(3) - 0.5)) * T_4_0_true
# T_5_0_init = SE2.exp(0.1 * 2 * (np.random.rand(3) - 0.5)) * T_5_0_true
# T_6_0_init = SE2.exp(0.1 * 2 * (np.random.rand(3) - 0.5)) * T_6_0_true

# Constant wrong start
offset1 = np.array([-0.1, 0.1, -0.1])
offset2 = np.array([0.1, -0.1, 0.1])
T_1_0_init = T_1_0_true
T_2_0_init = SE2.exp(offset1) * T_2_0_true
T_3_0_init = SE2.exp(offset2) * T_3_0_true
T_4_0_init = SE2.exp(offset1) * T_4_0_true
T_5_0_init = SE2.exp(offset2) * T_5_0_true
T_6_0_init = SE2.exp(offset1) * T_6_0_true


# Either we need a prior on the first pose, or it needs to be held constant
# so that the resulting system of linear equations is solveable
prior_stiffness = invsqrt(1e-12 * np.identity(3))
odom_stiffness = invsqrt(1e-3 * np.identity(3))
loop_stiffness = invsqrt(1. * np.identity(3))

cost0 = PoseCost(T_1_0_obs, prior_stiffness)
cost0_params = ['T_1_0']

cost1 = PoseToPoseCost(T_2_1_obs, odom_stiffness)
cost1_params = ['T_1_0', 'T_2_0']

cost2 = PoseToPoseCost(T_3_2_obs, odom_stiffness)
cost2_params = ['T_2_0', 'T_3_0']

cost3 = PoseToPoseCost(T_4_3_obs, odom_stiffness)
cost3_params = ['T_3_0', 'T_4_0']

cost4 = PoseToPoseCost(T_5_4_obs, odom_stiffness)
cost4_params = ['T_4_0', 'T_5_0']

cost5 = PoseToPoseCost(T_6_5_obs, odom_stiffness)
cost5_params = ['T_5_0', 'T_6_0']

cost6 = PoseToPoseCost(T_6_2_obs, loop_stiffness)
cost6_params = ['T_2_0', 'T_6_0']

options = Options()
options.allow_nondecreasing_steps = True
options.max_nondecreasing_steps = 3

problem = Problem(options)
problem.add_residual_block(cost0, cost0_params)
problem.add_residual_block(cost1, cost1_params)
problem.add_residual_block(cost2, cost2_params)
problem.add_residual_block(cost3, cost3_params)
problem.add_residual_block(cost4, cost4_params)
problem.add_residual_block(cost5, cost5_params)
problem.add_residual_block(cost6, cost6_params)

# problem.set_parameters_constant('T_1_0')
# problem.set_parameters_constant('T_3_0')

params_init = {'T_1_0': T_1_0_init, 'T_2_0': T_2_0_init,
               'T_3_0': T_3_0_init, 'T_4_0': T_4_0_init,
               'T_5_0': T_5_0_init, 'T_6_0': T_6_0_init}
params_true = {'T_1_0': T_1_0_true, 'T_2_0': T_2_0_true,
               'T_3_0': T_3_0_true, 'T_4_0': T_4_0_true,
               'T_5_0': T_5_0_true, 'T_6_0': T_6_0_true}

problem.initialize_params(params_init)

print("Initial Cost: %e\n" % problem.eval_cost())

params_final = problem.solve()
print()

print("Final Cost: %e\n" % problem.eval_cost())

print("Initial Error:")
for key in params_true.keys():
    print('{}: {}'.format(key, SE2.log(
        params_init[key].inv() * params_true[key])))

print()

print("Final Error:")
for key in params_true.keys():
    print('{}: {}'.format(key, SE2.log(
        params_final[key].inv() * params_true[key])))
