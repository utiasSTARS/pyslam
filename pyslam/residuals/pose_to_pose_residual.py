import numpy as np


class PoseToPoseResidual:
    """Binary pose-to-pose residual given relative pose mesurement in SE2/SE3."""

    def __init__(self, T_2_1_obs, stiffness):
        self.T_2_1_obs = T_2_1_obs
        self.stiffness = stiffness
        self.obstype = type(T_2_1_obs)

    def evaluate(self, params, compute_jacobians=None):
        T_1_0_est = params[0]
        T_2_0_est = params[1]

        residual = np.dot(self.stiffness,
                          self.obstype.log(
                              T_2_0_est.dot(T_1_0_est.inv().dot(self.T_2_1_obs.inv()))))

        if compute_jacobians:
            jacobians = [None for _ in enumerate(params)]

            if compute_jacobians[0]:
                jacobians[0] = np.dot(self.stiffness, -T_2_0_est.dot(T_1_0_est.inv()).adjoint())

            if compute_jacobians[1]:
                jacobians[1] = np.dot(
                    self.stiffness, np.identity(self.obstype.dof))

            return residual, jacobians

        return residual
