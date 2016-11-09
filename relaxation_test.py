import numpy as np

from costs import *
from liegroups import SE3, SO3
from problem import Options, Problem

T_1_0_obs = SE3.identity()
T_2_1_obs = SE3.exp(
    np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1]))
T_3_2_obs = SE3.exp(
    np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1]))

T_1_0_est = SE3.identity()
T_2_0_est = T_2_1_obs * T_1_0_obs
T_3_0_est = T_3_2_obs * T_2_1_obs * T_1_0_obs

T_1_0_est = SE3.exp(0.5 * 2 * (np.random.rand(6) - 0.5)) * T_1_0_est
T_2_0_est = SE3.exp(0.5 * 2 * (np.random.rand(6) - 0.5)) * T_2_0_est
T_3_0_est = SE3.exp(0.5 * 2 * (np.random.rand(6) - 0.5)) * T_3_0_est
# T_1_0_est = SE3.exp(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])) * T_1_0_est
# T_2_0_est = SE3.exp(np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])) * T_2_0_est
# T_3_0_est = SE3.exp(np.array([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1])) * T_3_0_est

# Either we need a prior on the first pose, or it needs to be held constant
# so that the resulting system of linear equations is solveable
cost0 = SE3Cost(T_1_0_obs, np.linalg.inv(1e-12 * np.identity(6)))
cost0_params = [T_1_0_est]

cost1 = SE3toSE3Cost(T_2_1_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost1_params = [T_1_0_est, T_2_0_est]

cost2 = SE3toSE3Cost(T_3_2_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost2_params = [T_2_0_est, T_3_0_est]

options = Options()
options.allow_nonmonotonic_steps = True

problem = Problem(options)
problem.add_residual_block(cost0, cost0_params)
problem.add_residual_block(cost1, cost1_params)
problem.add_residual_block(cost2, cost2_params)

# problem.set_parameter_blocks_constant(cost0_params)
# print(problem.constant_params)
# problem.set_parameter_blocks_variable(cost0_params)

problem.solve()

print("Final:")
for p in problem.param_list:
    print(p)
