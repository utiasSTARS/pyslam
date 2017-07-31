import numpy as np
from liegroups import SE3


class ReprojectionResidual:
    """Reprojection error for any kind of camera."""

    def __init__(self, camera, obs, stiffness):
        self.camera = camera
        self.obs = obs
        self.stiffness = stiffness

    def evaluate(self, params, compute_jacobians=None):
        T_cam_w = params[0]
        pt_w = params[1]
        pt_cam = T_cam_w.dot(pt_w)

        if compute_jacobians:
            jacobians = [None for _ in enumerate(params)]

            predicted_obs, cam_jacobian = self.camera.project(
                pt_cam, compute_jacobians=True)
            residual = np.dot(self.stiffness, predicted_obs - self.obs)

            if compute_jacobians[0]:
                jacobians[0] = np.dot(self.stiffness,
                                      cam_jacobian.dot(SE3.odot(pt_cam)))

            if compute_jacobians[1]:
                jacobians[1] = np.dot(self.stiffness,
                                      cam_jacobian.dot(T_cam_w.rot.as_matrix()))

            return residual, jacobians

        residual = np.dot(self.stiffness,
                          self.camera.project(pt_cam) - self.obs)
        return residual
