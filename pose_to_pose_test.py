import numpy as np

from costs import SE3toSE3Cost
from liegroups import SE3, SO3

T_1_0 = SE3.identity()
T_2_0 = SE3(SO3.identity(), np.array([0.1, 0, 0]))

T_2_1 = T_2_0 * T_1_0.inv()
obs_covar = 1e-3 * np.identity(6)

params = [T_1_0, T_2_0]
obs = T_2_1

cost = SE3toSE3Cost(obs, np.linalg.inv(obs_covar))
print(cost.evaluate(params, [True, True]))
