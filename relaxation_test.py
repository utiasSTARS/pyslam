import numpy as np

from costs import *
from liegroups import SE3, SO3
from problem import Problem

T_1_0_obs = SE3.identity()
T_2_1_obs = SE3.exp(
    np.array([0.1, -0.1, 0.1, 0, 0, 0]))
T_3_2_obs = SE3.exp(
    np.array([0.1, -0.1, 0.1, 0, 0, 0]))

T_1_0_est = SE3.identity()
T_2_0_est = T_2_1_obs * T_1_0_obs
T_3_0_est = T_3_2_obs * T_2_1_obs * T_1_0_obs

T_2_0_est = SE3.exp(0.02 * np.random.rand(6)) * T_2_0_est
T_3_0_est = SE3.exp(0.02 * np.random.rand(6)) * T_3_0_est
# T_2_0_est = SE3.exp(np.array([0.01, 0.01, 0.01, 0, 0, 0])) * T_2_0_est
# T_3_0_est = SE3.exp(np.array([0.01, 0.01, 0.01, 0, 0, 0])) * T_3_0_est

# Either we need a prior on the first pose, or it needs to be held constant
# so that the resulting system of linear equations is solveable
cost0 = SE3Cost(T_1_0_obs, np.linalg.inv(1e-12 * np.identity(6)))
cost0_params = [T_1_0_est]

cost1 = SE3toSE3Cost(T_2_1_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost1_params = [T_1_0_est, T_2_0_est]

cost2 = SE3toSE3Cost(T_3_2_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost2_params = [T_2_0_est, T_3_0_est]

problem = Problem()
problem.add_residual_block(cost0, cost0_params)
problem.add_residual_block(cost1, cost1_params)
problem.add_residual_block(cost2, cost2_params)
problem.solve()
