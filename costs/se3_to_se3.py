import numpy as np

from liegroups import SE3


class SE3toSE3Cost:
    """Pose-to-pose cost given relative pose mesurement in SE(3)."""

    def __init__(self, obs, weight):
        self.obs = obs
        self.weight = weight

    def evaluate(self, params, do_jacobians=[]):
        T_1_0 = params[0]
        T_2_0 = params[1]
        T_2_1_obs = self.obs

        T_1_2_est = T_1_0 * T_2_0.inv()

        residual = SE3.log(T_2_1_obs * T_1_2_est)

        jacobians = [[] for _ in range(len(params))]

        # do_jacobians is empty or an array of booleans with the same length
        # as params
        if do_jacobians:
            if do_jacobians[0]:
                jacobians[0] = -T_1_2_est.inv().adjoint()

            if do_jacobians[1]:
                jacobians[1] = np.identity(6)

        return residual, jacobians
