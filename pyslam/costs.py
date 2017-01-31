import numpy as np

from liegroups import SE3

from pyslam.utils import bilinear_interpolate


class QuadraticCost:
    """A simple quadratic cost for fitting a parabola to data."""

    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.weight = np.array(weight)

    def evaluate(self, params, compute_jacobians=None):
        residual = np.array([params[0] * self.x * self.x
                             + params[1] * self.x
                             + params[2]
                             - self.y])

        if compute_jacobians:
            jacobians = [None for _ in range(len(params))]

            if compute_jacobians[0]:
                jacobians[0] = self.x * self.x

            if compute_jacobians[1]:
                jacobians[1] = self.x

            if compute_jacobians[2]:
                jacobians[2] = 1.

            return residual, jacobians

        return residual


class PoseCost:
    """Unary pose cost given absolute pose measurement in SE2/SE3."""

    def __init__(self, T_obs, weight):
        self.T_obs = T_obs
        self.weight = weight
        self.obstype = type(T_obs)

    def evaluate(self, params, compute_jacobians=None):
        T_est = params[0]

        residual = self.obstype.log(T_est * self.T_obs.inv())

        if compute_jacobians:
            jacobians = [None for _ in range(len(params))]

            if compute_jacobians[0]:
                jacobians[0] = np.identity(self.obstype.dof)

            return residual, jacobians

        return residual


class PoseToPoseCost:
    """Binary pose-to-pose cost given relative pose mesurement in SE2/SE3."""

    def __init__(self, T_2_1_obs, weight):
        self.T_2_1_obs = T_2_1_obs
        self.weight = weight
        self.obstype = type(T_2_1_obs)

    def evaluate(self, params, compute_jacobians=None):
        T_1_0_est = params[0]
        T_2_0_est = params[1]

        residual = self.obstype.log(
            T_2_0_est * T_1_0_est.inv() * self.T_2_1_obs.inv())

        if compute_jacobians:
            jacobians = [None for _ in range(len(params))]

            if compute_jacobians[0]:
                jacobians[0] = -T_2_0_est.adjoint()

            if compute_jacobians[1]:
                jacobians[1] = np.identity(self.obstype.dof)

            return residual, jacobians

        return residual


class ReprojectionCost:
    """Reprojection error for any kind of camera."""

    def __init__(self, camera, obs, weight):
        self.camera = camera
        self.obs = obs
        self.weight = weight

    def evaluate(self, params, compute_jacobians=None):
        T_cam_w = params[0]
        pt_w = params[1]
        pt_cam = T_cam_w * pt_w

        if compute_jacobians:
            jacobians = [None for _ in range(len(params))]

            predicted_obs, cam_jacobian = self.camera.project(
                pt_cam, compute_jacobians=True)
            residual = predicted_obs - self.obs

            if compute_jacobians[0]:
                jacobians[0] = cam_jacobian.dot(SE3.odot(pt_cam))

            if compute_jacobians[1]:
                jacobians[1] = cam_jacobian.dot(T_cam_w.rot.as_matrix())

            return residual, jacobians

        residual = self.camera.project(pt_cam) - self.obs
        return residual


class PhotometricCost:
    """Photometric cost for greyscale images."""

    def __init__(self, camera, im_ref, disp_ref, im_track, jac_ref, weight):
        self.camera = camera
        self.im_ref = im_ref
        self.disp_ref = disp_ref
        self.im_track = im_track
        self.jac_ref = jac_ref
        self.weight = weight
        self.u, self.v = np.meshgrid(list(range(0, camera.w)),
                                     list(range(0, camera.h)),
                                     indexing='xy')

    def evaluate(self, params, compute_jacobians=None):
        T_track_ref = params[0]

        uvd_ref = np.array([self.u.flatten(), self.v.flatten(),
                            self.disp_ref.flatten()]).T
        im_ref_true = self.im_ref.flatten()

        # Filter out bad measurements (NaN disparity)
        valid_ref = self.camera.is_valid_measurement(uvd_ref)
        uvd_ref = uvd_ref[valid_ref, :]
        im_ref_true = im_ref_true[valid_ref]

        # Reproject reference image pixels into tracking image to predict the
        # reference image based on the tracking image
        pt_ref = self.camera.triangulate(np.array(uvd_ref))
        pt_track = T_track_ref * pt_ref
        uvd_track = self.camera.project(pt_track)

        # Filter out bad measurements (out of bounds coordinates, nonpositive
        # disparity)
        valid_track = self.camera.is_valid_measurement(uvd_track)
        uvd_track = uvd_track[valid_track, :]
        uvd_ref = uvd_ref[valid_track]
        im_ref_true = im_ref_true[valid_track]

        # The residual is the intensity difference between the estimated
        # reference image pixels and the true reference image pixels
        im_ref_est = bilinear_interpolate(
            self.im_track, uvd_track[:, 0], uvd_track[:, 1])
        residual = im_ref_est - im_ref_true

        # DEBUG: Rebuild the residual image as a sanity check
        # residual_image = np.empty(self.im_ref.shape)
        # residual_image.fill(np.nan)
        # residual_image[uvd_ref.astype(int)[:, 1],
        #                uvd_ref.astype(int)[:, 0]] = residual
        # return residual_image

        return residual
