import numpy as np
from liegroups import SE3

class ReprojectionCostFrameToFrame:
    """Frame to frame reprojection error for any kind of camera."""

    def __init__(self, camera, obs_1, obs_2, stiffness):
        self.camera = camera
        self.obs_1 = obs_1
        self.obs_2 = obs_2
        self.stiffness = stiffness

    def evaluate(self, params, compute_jacobians=None):
        """ This is my docstring. """
        T_cam2_cam1 = params[0]
        pt_cam1 = self.camera.triangulate(self.obs_1)
        pt_cam2 = T_cam2_cam1 * pt_cam1

        if compute_jacobians:
            jacobians = [None for _ in enumerate(params)]

            predicted_obs, cam_jacobian = self.camera.project(
                pt_cam2, compute_jacobians=True)

            residual = np.dot(self.stiffness, predicted_obs - self.obs_2)

            if compute_jacobians[0]:
                jacobians[0] = np.dot(self.stiffness,
                                      cam_jacobian.dot(SE3.odot(pt_cam2)))

            return residual, jacobians

        residual = np.dot(self.stiffness,
                          self.camera.project(pt_cam2) - self.obs_2)
        return residual
