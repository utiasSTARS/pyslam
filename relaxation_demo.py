import copy

import numpy as np

from costs import *
from liegroups import SE3, SO3
from problem import Options, Problem

T_1_0_true = SE3.identity()
T_2_0_true = SE3(SO3.identity(), -np.array([0.5, 0, 0]))
T_3_0_true = SE3(SO3.identity(), -np.array([1, 0, 0]))
T_4_0_true = SE3(SO3.rotz(np.pi / 2),
                 -(SO3.rotz(np.pi / 2) * np.array([1, 0.5, 0])))
T_5_0_true = SE3(SO3.rotz(np.pi), -(SO3.rotz(np.pi) * np.array([0.5, 0.5, 0])))
T_6_0_true = SE3(SO3.rotz(3 * np.pi / 2),
                 -(SO3.rotz(3 * np.pi / 2) * np.array([0.5, 0, 0])))

# Odometry
T_1_0_obs = SE3.identity()
T_2_1_obs = T_2_0_true * T_1_0_true.inv()
T_3_2_obs = T_3_0_true * T_2_0_true.inv()
T_4_3_obs = T_4_0_true * T_3_0_true.inv()
T_5_4_obs = T_5_0_true * T_4_0_true.inv()
T_6_5_obs = T_6_0_true * T_5_0_true.inv()

# Loop closure
T_6_2_obs = T_6_0_true * T_2_0_true.inv()

# Random start
T_1_0_est = SE3.exp(0.1 * 2 * (np.random.rand(6) - 0.5)) * T_1_0_true
T_2_0_est = SE3.exp(0.1 * 2 * (np.random.rand(6) - 0.5)) * T_2_0_true
T_3_0_est = SE3.exp(0.1 * 2 * (np.random.rand(6) - 0.5)) * T_3_0_true
T_4_0_est = SE3.exp(0.1 * 2 * (np.random.rand(6) - 0.5)) * T_4_0_true
T_5_0_est = SE3.exp(0.1 * 2 * (np.random.rand(6) - 0.5)) * T_5_0_true
T_6_0_est = SE3.exp(0.1 * 2 * (np.random.rand(6) - 0.5)) * T_6_0_true

# Constant wrong start
# T_1_0_est = SE3.exp(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])) * T_1_0_true
# T_2_0_est = SE3.exp(np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])) * T_2_0_true
# T_3_0_est = SE3.exp(np.array([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1])) * T_3_0_true

# Either we need a prior on the first pose, or it needs to be held constant
# so that the resulting system of linear equations is solveable
cost0 = SE3Cost(T_1_0_obs, np.linalg.inv(1e-12 * np.identity(6)))
cost0_params = [T_1_0_est]

cost1 = SE3toSE3Cost(T_2_1_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost1_params = [T_1_0_est, T_2_0_est]

cost2 = SE3toSE3Cost(T_3_2_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost2_params = [T_2_0_est, T_3_0_est]

cost3 = SE3toSE3Cost(T_4_3_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost3_params = [T_3_0_est, T_4_0_est]

cost4 = SE3toSE3Cost(T_5_4_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost4_params = [T_4_0_est, T_5_0_est]

cost5 = SE3toSE3Cost(T_6_5_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost5_params = [T_5_0_est, T_6_0_est]

cost6 = SE3toSE3Cost(T_6_2_obs, np.linalg.inv(1e-3 * np.identity(6)))
cost6_params = [T_2_0_est, T_6_0_est]

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

# problem.set_parameters_constant(cost0_params)
# problem.set_parameters_variable(cost0_params)

params = [T_1_0_est, T_2_0_est, T_3_0_est, T_4_0_est, T_5_0_est, T_6_0_est]
true_params = [T_1_0_true, T_2_0_true, T_3_0_true,
               T_4_0_true, T_5_0_true, T_6_0_true]

print("Initial:")
for p in params:
    print(p)
print()

problem.solve()
print()

print("Final:")
for p in params:
    print(p)
print()

print("Final Error:")
for est, true in zip(params, true_params):
    print(SE3.log(est.inv() * true))
