import numpy as np

from liegroups import SE3
from pyslam.utils import stackmul

from numba import guvectorize, float64


SE3_ODOT_SHAPE = np.empty(6)


@guvectorize([(float64[:], float64[:], float64[:, :])],
             '(n),(m)->(n,m)', nopython=True, cache=True, target='parallel')
def fast_se3_odot(vec, junk, out):
    out[0, 0] = 1.
    out[0, 1] = 0.
    out[0, 2] = 0.
    out[0, 3] = 0.
    out[0, 4] = vec[2]
    out[0, 5] = -vec[1]
    out[1, 0] = 0.
    out[1, 1] = 1.
    out[1, 2] = 0.
    out[1, 3] = -vec[2]
    out[1, 4] = 0.
    out[1, 5] = vec[0]
    out[2, 0] = 0.
    out[2, 1] = 0.
    out[2, 2] = 1.
    out[2, 3] = vec[1]
    out[2, 4] = -vec[0]
    out[2, 5] = 0.


class ReprojectionMotionOnlyResidual:
    """Frame to frame reprojection error for any kind of camera."""

    def __init__(self, camera, obs_1, obs_2, stiffness):
        self.camera = camera
        self.obs_1 = obs_1
        self.obs_2 = obs_2
        self.stiffness = stiffness

        self.pt_1 = self.camera.triangulate(self.obs_1)

    def evaluate(self, params, compute_jacobians=None):
        """ This is my docstring. """
        T_2_1 = params[0]
        pt_2 = T_2_1.dot(self.pt_1)

        if compute_jacobians:
            jacobians = [None for _ in enumerate(params)]

            obs_2_pred, cam_jacobian = self.camera.project(
                pt_2, compute_jacobians=True)

            residual = np.dot(self.stiffness, obs_2_pred - self.obs_2)

            if compute_jacobians[0]:
                jacobians[0] = np.dot(self.stiffness,
                                      cam_jacobian.dot(SE3.odot(pt_2)))

            return residual, jacobians

        residual = np.dot(self.stiffness,
                          self.camera.project(pt_2) - self.obs_2)
        return residual


class ReprojectionMotionOnlyBatchResidual:
    """Frame to frame reprojection error with batch jacobians (for multiple reprojections)."""

    def __init__(self, camera, obs_1, obs_2, stiffness):
        self.camera = camera
        self.obs_1 = obs_1
        self.obs_2 = obs_2
        self.stiffness = stiffness

        self.pts_1 = self.camera.triangulate(self.obs_1)
        self.num_pts = self.pts_1.shape[0]

    def evaluate(self, params, compute_jacobians=None):
        T_2_1 = params[0]
        pts_2 = T_2_1.dot(self.pts_1)

        if compute_jacobians:
            jacobians = [None for _ in enumerate(params)]

            obs_2_pred, cam_proj_jac = self.camera.project(
                pts_2, compute_jacobians=True)

            residual = self.stiffness.dot((obs_2_pred - self.obs_2).T).T
            residual = residual.reshape(3*self.num_pts)

            if compute_jacobians[0]:
                odot_pts_2 = fast_se3_odot(pts_2, SE3_ODOT_SHAPE)
                jac = stackmul(cam_proj_jac, odot_pts_2)

                stiffness = np.broadcast_to(self.stiffness,
                                            (self.num_pts, 3, 3))
                jac = stackmul(stiffness, jac)

                # Reshape back into a (3*N, 6) Jacobian
                jacobians[0] = np.reshape(jac, [3*self.num_pts, 6])

            return residual, jacobians

        # Multiply (3,3) by (3,N), and then reshape to get a (3*N,) array
        obs_2_pred = self.camera.project(pts_2)
        residual = self.stiffness.dot((obs_2_pred - self.obs_2).T).T
        residual = residual.reshape(3*self.num_pts)

        return residual
