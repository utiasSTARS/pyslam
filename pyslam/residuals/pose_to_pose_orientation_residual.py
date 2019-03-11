import numpy as np


class PoseToPoseOrientationResidual:
    """Binary pose-to-pose residual given relative rotation mesurement in SO3."""

    def __init__(self, C_2_1_obs, stiffness):
        self.C_2_1_obs = C_2_1_obs
        self.stiffness = stiffness
        self.obstype = type(C_2_1_obs)

    def evaluate(self, params, compute_jacobians=None):
        T_1_0_est = params[0]
        T_2_0_est = params[1]

        #TODO: Make this more general for SE(2) poses
        P2 = np.zeros((3, 6))
        P2[:, 3:] = np.eye(3)
        P1 = np.zeros((3, 6))
        P1[:, 3:] = T_2_0_est.dot(T_1_0_est.inv()).rot.as_matrix()

        residual = np.dot(self.stiffness,
                              self.obstype.log(
                                  T_2_0_est.dot(T_1_0_est.inv()).rot.dot(self.C_2_1_obs.inv())
                              )
                          )
        if compute_jacobians:
            jacobians = [None for _ in enumerate(params)]

            if compute_jacobians[0]:
                jacobians[0] = np.dot(self.stiffness, -P1)

            if compute_jacobians[1]:
                jacobians[1] = np.dot(self.stiffness, P2)

            return residual, jacobians

        return residual
