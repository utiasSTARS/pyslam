import numpy as np

from liegroups import SE3


class SE3Cost:
    """Unary pose cost given absolute pose measurement in SE(3)."""

    def __init__(self, T_obs, weight):
        self.T_obs = T_obs
        self.weight = weight

    def evaluate(self, params, compute_jacobians=None):
        T_est = params[0]

        residual = SE3.log(T_est * self.T_obs.inv())

        if compute_jacobians:
            jacobians = [None for _ in range(len(params))]

            if compute_jacobians[0]:
                jacobians[0] = np.identity(6)

            return residual, jacobians

        return residual


class SE3toSE3Cost:
    """Binary pose-to-pose cost given relative pose mesurement in SE(3)."""

    def __init__(self, T_2_1_obs, weight):
        self.T_2_1_obs = T_2_1_obs
        self.weight = weight

    def evaluate(self, params, compute_jacobians=None):
        T_1_0_est = params[0]
        T_2_0_est = params[1]

        residual = SE3.log(T_2_0_est * T_1_0_est.inv() * self.T_2_1_obs.inv())

        if compute_jacobians:
            jacobians = [None for _ in range(len(params))]

            if compute_jacobians[0]:
                jacobians[0] = -T_2_0_est.adjoint()

            if compute_jacobians[1]:
                jacobians[1] = np.identity(6)

            return residual, jacobians

        return residual
