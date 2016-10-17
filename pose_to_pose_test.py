import numpy as np

from costs import *
from problem import Problem
from liegroups import SE3, SO3

T_1_0 = SE3.identity()
T_2_0 = SE3(SO3.identity(), np.array([0.13, 0.0, 0.0]))
T_3_0 = SE3(SO3.identity(), np.array([0.16, 0.0, 0.0]))

T_1_0_obs = SE3.identity()
T_2_1_obs = SE3(SO3.identity(), np.array([0.1, 0, 0]))
T_3_2_obs = SE3(SO3.identity(), np.array([0.1, 0, 0]))

# Either we need a prior on the first pose, or it needs to be held constant
# so that the resulting system of linear equations is solveable
cost0 = SE3Cost(T_1_0_obs, np.linalg.inv(1e-12 * np.identity(6)))
cost0_params = [T_1_0]

cost1 = SE3toSE3Cost(T_2_1_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost1_params = [T_1_0, T_2_0]

cost2 = SE3toSE3Cost(T_3_2_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost2_params = [T_2_0, T_3_0]

problem = Problem()
problem.add_residual_block(cost0, cost0_params)
problem.add_residual_block(cost1, cost1_params)
problem.add_residual_block(cost2, cost2_params)
problem.solve()
