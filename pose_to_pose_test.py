import numpy as np

from costs import *
from problem import Problem
from liegroups import SE3, SO3

T_1_0 = SE3.identity()
T_2_0 = SE3(SO3.identity(), np.array([0.1, 0, 0]))

T_1_0_obs = SE3.identity()
T_2_1_obs = T_2_0 * T_1_0.inv()


cost0 = SE3Cost(T_1_0_obs, np.linalg.inv(1e-12 * np.identity(6)))
cost0_params = [T_1_0]

cost1 = SE3toSE3Cost(T_2_1_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost1_params = [T_1_0, T_2_0]

problem = Problem()
problem.add_residual_block(cost0, cost0_params)
problem.add_residual_block(cost1, cost1_params)
problem.solve()
