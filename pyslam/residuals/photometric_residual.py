import numpy as np
import scipy.interpolate
import time

from liegroups import SE3
from pyslam.utils import stackmul, bilinear_interpolate

from numba import guvectorize, float32, float64


SE3_ODOT_SHAPE = np.empty(6)


@guvectorize([(float32[:], float32[:, :]),
              (float64[:], float64[:, :])],
             '(n)->(n,n)', nopython=True, cache=True, target='parallel')
def fast_neg_so3_wedge(vec, out):
    out[0, 0] = 0.
    out[0, 1] = vec[2]
    out[0, 2] = -vec[1]
    out[1, 0] = -vec[2]
    out[1, 1] = 0.
    out[1, 2] = vec[0]
    out[2, 0] = vec[1]
    out[2, 1] = -vec[0]
    out[2, 2] = 0.


@guvectorize([(float32[:], float32[:], float32[:, :]),
              (float64[:], float64[:], float64[:, :])],
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


class PhotometricResidualSO3:
    """SO3 (homography) photometric residual for greyscale images.
    Uses the pre-computed reference image jacobian as an approximation to the
    tracking image jacobian under the assumption that the camera motion is small."""

    def __init__(self, camera, im_ref, im_track, im_jac,
                 intensity_stiffness, min_grad=0.):
        self.camera = camera
        self.im_ref = im_ref.ravel()

        u_range = range(0, self.camera.w)
        v_range = range(0, self.camera.h)
        u_coords, v_coords = np.meshgrid(u_range, v_range, indexing='xy')
        self.uv_ref = np.vstack([u_coords.ravel(),
                                 v_coords.ravel(),
                                 np.ones(u_coords.size)]).T
        self.im_jac = np.vstack([im_jac[0].ravel(),
                                 im_jac[1].ravel(),
                                 np.zeros(im_jac[0].size)]).T

        self.im_track = im_track
        self.intensity_stiffness = intensity_stiffness
        self.min_grad = min_grad

        # Filter out pixels with weak gradients
        grad_ref = np.linalg.norm(self.im_jac, axis=1)
        strong_pixels = grad_ref >= self.min_grad
        self.uv_ref = self.uv_ref.compress(strong_pixels, axis=0)
        self.im_ref = self.im_ref.compress(strong_pixels)
        self.im_jac = self.im_jac.compress(strong_pixels, axis=0)

        # Precompute reference points
        self.pt_ref = self.camera.invK.dot(self.uv_ref.T).T

    def evaluate(self, params, compute_jacobians=None):
        R_track_ref = params[0]
        pt_track = R_track_ref.dot(self.pt_ref)
        uv_track = self.camera.K.dot(pt_track.T).T

        im_ref_est = bilinear_interpolate(self.im_track,
                                          uv_track[:, 0],
                                          uv_track[:, 1])

        residual = self.intensity_stiffness * (im_ref_est - self.im_ref)

        if compute_jacobians:
            jacobians = [None]

            if compute_jacobians[0]:
                temp = stackmul(self.camera.K, fast_neg_so3_wedge(pt_track))
                jacobians[0] = self.intensity_stiffness * \
                    np.squeeze(stackmul(self.im_jac[:, np.newaxis, :], temp))

            return residual, jacobians

        return residual


class PhotometricResidualSE3:
    """Full SE3 photometric residual for greyscale images.
    Uses the pre-computed reference image jacobian as an approximation to the
    tracking image jacobian under the assumption that the camera motion is small."""

    def __init__(self, camera, im_ref, disp_ref, im_track, im_jac,
                 intensity_stiffness, disparity_stiffness, min_grad=0.):
        self.camera = camera
        self.im_ref = im_ref.ravel()

        u_range = range(0, self.camera.w)
        v_range = range(0, self.camera.h)
        u_coords, v_coords = np.meshgrid(u_range, v_range, indexing='xy')
        self.uvd_ref = np.vstack([u_coords.ravel(),
                                  v_coords.ravel(),
                                  disp_ref.ravel()]).T
        self.im_jac = np.vstack([im_jac[0].ravel(),
                                 im_jac[1].ravel()]).T

        self.im_track = im_track

        self.intensity_stiffness = intensity_stiffness
        self.disparity_stiffness = disparity_stiffness
        self.intensity_covar = intensity_stiffness ** -2
        self.disparity_covar = disparity_stiffness ** -2

        self.min_grad = min_grad

        # Filter out invalid pixels (NaN or negative disparity)
        valid_pixels = self.camera.is_valid_measurement(self.uvd_ref)
        self.uvd_ref = self.uvd_ref.compress(valid_pixels, axis=0)
        self.im_ref = self.im_ref.compress(valid_pixels)
        self.im_jac = self.im_jac.compress(valid_pixels, axis=0)

        # Filter out pixels with weak gradients
        grad_ref = np.linalg.norm(self.im_jac, axis=1)
        strong_pixels = grad_ref >= self.min_grad
        self.uvd_ref = self.uvd_ref.compress(strong_pixels, axis=0)
        self.im_ref = self.im_ref.compress(strong_pixels)
        self.im_jac = self.im_jac.compress(strong_pixels, axis=0)

        # Precompute triangulated 3D points
        self.pt_ref, self.triang_jac = self.camera.triangulate(
            self.uvd_ref, compute_jacobians=True)

    def evaluate(self, params, compute_jacobians=None):
        if len(params) == 1:
            T_track_ref = params[0]
        elif len(params) == 2:
            T_track_ref = SE3(params[0], params[1])
        else:
            raise ValueError(
                'In PhotometricResidual.evaluate() params must have length 1 or 2')

        # Reproject reference image pixels into tracking image to predict the
        # reference image based on the tracking image
        pt_track = T_track_ref * self.pt_ref
        uvd_track, project_jac = self.camera.project(
            pt_track, compute_jacobians=True)

        # Filter out bad measurements
        # where returns a tuple
        valid_pixels = self.camera.is_valid_measurement(uvd_track)

        # The residual is the intensity difference between the estimated
        # reference image pixels and the true reference image pixels
        # This is actually faster than filtering the intermediate results!
        im_ref_est = bilinear_interpolate(self.im_track,
                                          uvd_track[:, 0],
                                          uvd_track[:, 1])
        residual = (im_ref_est - self.im_ref).compress(valid_pixels)

        # We need the jacobian of the residual w.r.t. the disparity
        # to compute a reasonable stiffness paramater
        # This is actually faster than filtering the intermediate results!
        im_proj_jac = stackmul(self.im_jac[:, np.newaxis, :],
                               project_jac[:, 0:2, :])  # Nx1x3
        temp = stackmul(im_proj_jac, T_track_ref.rot.as_matrix())  # Nx1x3
        im_disp_jac = stackmul(temp, self.triang_jac[:, :, 2:3])  # Nx1x1

        # Compute the overall stiffness
        # \sigma^2 = \sigma^2_I + J_d \sigma^2_d J_d^T
        stiffness = 1. / np.sqrt(self.intensity_covar +
                                 self.disparity_covar * np.squeeze(
                                     im_disp_jac.compress(valid_pixels, axis=0))**2)
        # stiffness = self.intensity_stiffness
        residual = stiffness * residual

        # DEBUG: Rebuild residual and disparity images
        # self._rebuild_images(residual, im_ref_est, self.im_ref, valid_pixels)
        # import ipdb
        # ipdb.set_trace()

        # Jacobian time!
        if compute_jacobians:
            if any(compute_jacobians):
                # This is actually faster than filtering the intermediate
                # results!
                odot_pt_track = fast_se3_odot(pt_track, SE3_ODOT_SHAPE)
                jac = stackmul(im_proj_jac, odot_pt_track).compress(
                    valid_pixels, axis=0)
                jac = (stiffness * np.squeeze(jac.T)).T

            if len(params) == 1:
                # SE3 case
                jacobians = [None]

                if compute_jacobians[0]:
                    jacobians[0] = jac

            elif len(params) == 2:
                # (SO3, t) case
                jacobians = [None, None]

                if compute_jacobians[0]:
                    # Rotation part
                    jacobians[0] = jac[:, 3:6]
                if compute_jacobians[1]:
                    # Translation part
                    jacobians[1] = jac[:, 0:3]

            return residual, jacobians

        return residual

    def _rebuild_images(self, residual, im_ref_est, im_ref_true, valid_pixels):
        """Debug function to rebuild the filtered
        residual and disparity images as a sanity check"""
        uvd_ref = self.uvd_ref[valid_pixels]
        imshape = (self.camera.h, self.camera.w)

        self.actual_reference_image = np.full(imshape, np.nan)
        self.actual_reference_image[
            uvd_ref.astype(int)[:, 1],
            uvd_ref.astype(int)[:, 0]] = im_ref_true[valid_pixels]

        self.estimated_reference_image = np.full(imshape, np.nan)
        self.estimated_reference_image[
            uvd_ref.astype(int)[:, 1],
            uvd_ref.astype(int)[:, 0]] = im_ref_est[valid_pixels]

        self.residual_image = np.full(imshape, np.nan)
        self.residual_image[
            uvd_ref.astype(int)[:, 1],
            uvd_ref.astype(int)[:, 0]] = residual

        self.disparity_image = np.full(imshape, np.nan)
        self.disparity_image[
            uvd_ref.astype(int)[:, 1],
            uvd_ref.astype(int)[:, 0]] = uvd_ref[:, 2]
