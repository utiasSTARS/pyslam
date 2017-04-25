import numpy as np
import scipy.interpolate
import time

from pyslam.utils import stackmul, bilinear_interpolate

from numba import guvectorize, float64


se3_odot_shape = np.empty(6)


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


class PhotometricResidual:
    """Photometric residual for greyscale images.
    Uses the pre-computed reference image jacobian as an approximation to the
    tracking image jacobian under the assumption that the camera motion is small."""

    def __init__(self, camera, im_ref, disp_ref, jac_ref,
                 im_track, stiffness):
        self.camera = camera
        self.im_ref = im_ref.ravel()

        u_range = range(0, self.camera.w)
        v_range = range(0, self.camera.h)
        u_coords, v_coords = np.meshgrid(u_range, v_range, indexing='xy')
        self.uvd_ref = np.vstack(
            [u_coords.ravel(), v_coords.ravel(), disp_ref.ravel()]).T
        self.jac_ref = np.vstack([jac_ref[0, :, :].ravel(),
                                  jac_ref[1, :, :].ravel()]).T

        self.im_track = im_track

        self.stiffness = stiffness

        # Filter out invalid pixels (NaN or negative disparity)
        valid_pixels = np.where(
            self.camera.is_valid_measurement(
                self.uvd_ref))[0]  # where returns a tuple
        self.uvd_ref = self.uvd_ref[valid_pixels, :]
        self.im_ref = self.im_ref[valid_pixels]
        self.jac_ref = self.jac_ref[valid_pixels, :]

        # Filter out pixels with weak gradients
        grad_ref = np.linalg.norm(self.jac_ref, axis=1)
        strong_pixels = np.where(grad_ref >= 0.05)[0]  # where returns a tuple
        self.uvd_ref = self.uvd_ref[strong_pixels, :]
        self.im_ref = self.im_ref[strong_pixels]
        self.jac_ref = self.jac_ref[strong_pixels, :]

        # Precompute triangulated 3D points
        self.pt_ref, self.triang_jac = self.camera.triangulate(
            np.array(self.uvd_ref), compute_jacobians=True)

    def evaluate(self, params, compute_jacobians=None):
        T_track_ref = params[0]

        # Reproject reference image pixels into tracking image to predict the
        # reference image based on the tracking image
        # start = time.perf_counter()
        pt_track = T_track_ref * self.pt_ref
        # end = time.perf_counter()
        # print('\ntransform | {}'.format(end - start))

        # start = time.perf_counter()
        uvd_track, project_jac = self.camera.project(
            pt_track, compute_jacobians=True)
        # end = time.perf_counter()
        # print('project | {}'.format(end - start))

        # Filter out bad measurements
        # start = time.perf_counter()
        # where returns a tuple
        valid_pixels = np.where(self.camera.is_valid_measurement(uvd_track))[0]
        # end = time.perf_counter()
        # print('validate | {}'.format(end - start))

        # The residual is the intensity difference between the estimated
        # reference image pixels and the true reference image pixels
        # start = time.perf_counter()
        # This is actually faster than filtering the intermediate results!
        im_ref_est = bilinear_interpolate(self.im_track,
                                          uvd_track[:, 0],
                                          uvd_track[:, 1])
        residual = (im_ref_est - self.im_ref)[valid_pixels]
        # end = time.perf_counter()
        # print('interpolate residual | {}'.format(end - start))

        # We need the jacobian of the residual w.r.t. the disparity
        # to compute a reasonable stiffness paramater
        # start = time.perf_counter()
        # This is actually faster than filtering the intermediate results!
        im_proj_jac = stackmul(self.jac_ref[:, np.newaxis, :],
                               project_jac[:, 0:2, :])  # Nx1x3
        temp = stackmul(im_proj_jac, T_track_ref.rot.as_matrix())  # Nx1x3
        im_disp_jac = stackmul(temp, self.triang_jac[:, :, 2:3])  # Nx1x1

        # Compute the overall stiffness
        stiffness = 1. / np.sqrt(self.stiffness ** -
                                 2 + 4. * np.squeeze(
                                     im_disp_jac[valid_pixels, :, :])**2)

        residual = stiffness * residual
        # end = time.perf_counter()
        # print('residual stiffness | {}'.format(end - start))

        # DEBUG: Rebuild residual and disparity images
        # self._rebuild_images(residual, im_ref_est, self.im_ref, valid_pixels)
        # import ipdb
        # ipdb.set_trace()

        # Jacobian time!
        if compute_jacobians:
            jacobians = [None]

            if compute_jacobians[0]:
                # start = time.perf_counter()
                odot_pt_track = fast_se3_odot(pt_track, se3_odot_shape)
                # end = time.perf_counter()
                # print('jacobians odot | {}'.format(end - start))

                # start = time.perf_counter()
                # This is actually faster than filtering the intermediate
                # results!
                jacobians[0] = stackmul(im_proj_jac, odot_pt_track)[
                    valid_pixels, :, :]
                # end = time.perf_counter()
                # print('jacobians matmul | {}'.format(end - start))

                # start = time.perf_counter()
                # Transposes needed for proper broadcasting
                jacobians[0] = (stiffness * np.squeeze(jacobians[0].T)).T

                # end = time.perf_counter()
                # print('jacobians stiffness | {}\n'.format(end - start))

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
