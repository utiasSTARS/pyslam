import numpy as np
import scipy.interpolate
import time

from liegroups import SE3
from pyslam.utils import stackmul, bilinear_interpolate


class PhotometricResidual:
    """Photometric residual for greyscale images.
    Uses the pre-computed reference image jacobian as an approximation to the
    tracking image jacobian under the assumption that the camera motion is small."""

    def __init__(self, camera, im_ref, disp_ref, jac_ref,
                 im_track, stiffness):
        self.camera = camera
        self.im_ref = im_ref.flatten()

        u_range = range(0, self.camera.w)
        v_range = range(0, self.camera.h)
        u_coords, v_coords = np.meshgrid(u_range, v_range, indexing='xy')
        self.uvd_ref = np.vstack(
            [u_coords.flatten(), v_coords.flatten(), disp_ref.flatten()]).T
        self.jac_ref = np.vstack([jac_ref[0, :, :].flatten(),
                                  jac_ref[1, :, :].flatten()]).T

        self.im_track = im_track

        self.stiffness = stiffness

        # Filter out invalid pixels (NaN or negative disparity)
        valid_pixels = self.camera.is_valid_measurement(self.uvd_ref)
        self.uvd_ref = self.uvd_ref[valid_pixels, :]
        self.im_ref = self.im_ref[valid_pixels]
        self.jac_ref = self.jac_ref[valid_pixels, :]

        # Filter out pixels with weak gradients
        grad_ref = np.sum(self.jac_ref**2, axis=1)
        strong_pixels = grad_ref >= 0.25**2
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
        im_ref_true = self.im_ref

        pt_track = T_track_ref * self.pt_ref

        uvd_track, project_jac = self.camera.project(
            pt_track, compute_jacobians=True)

        # Filter out bad measurements
        valid_track = self.camera.is_valid_measurement(uvd_track)
        uvd_track = uvd_track[valid_track, :]
        pt_track = pt_track[valid_track, :]
        im_ref_true = im_ref_true[valid_track]
        im_jac = self.jac_ref[valid_track, :]
        project_jac = project_jac[valid_track, :, :]
        triang_jac = self.triang_jac[valid_track, :, :]

        # The residual is the intensity difference between the estimated
        # reference image pixels and the true reference image pixels
        im_ref_est = bilinear_interpolate(
            self.im_track, uvd_track[:, 0], uvd_track[:, 1])
        residual = im_ref_est - im_ref_true

        # We need the jacobian of the residual w.r.t. the disparity
        # to compute a reasonable stiffness paramater
        im_proj_jac = stackmul(
            im_jac[:, None, :], project_jac[:, 0:2, :])  # Nx1x3
        temp = stackmul(im_proj_jac, T_track_ref.rot.as_matrix())  # Nx1x3
        im_disp_jac = stackmul(temp, triang_jac[:, :, 2:3])  # Nx1x1

        # Compute the overall stiffness
        stiffness = 1. / np.sqrt(self.stiffness ** -
                                 2 + 4. * np.squeeze(im_disp_jac)**2)

        residual = stiffness * residual

        # DEBUG: Rebuild residual and disparity images
        # self._rebuild_images(residual, im_ref_est, im_ref_true, valid_track)
        # import ipdb
        # ipdb.set_trace()

        # Jacobian time!
        if compute_jacobians:
            jacobians = [None for _ in enumerate(params)]

            if compute_jacobians[0]:
                jacobians[0] = stackmul(im_proj_jac, SE3.odot(pt_track))
                jacobians[0] = (stiffness * np.squeeze(jacobians[0]).T).T

            return residual, jacobians

        return residual

    def _rebuild_images(self, residual, im_ref_est, im_ref_true, valid_track):
        """Debug function to rebuild the filtered
        residual and disparity images as a sanity check"""
        uvd_ref = self.uvd_ref[valid_track]
        imshape = (self.camera.h, self.camera.w)

        self.actual_reference_image = np.full(imshape, np.nan)
        self.actual_reference_image[uvd_ref.astype(int)[:, 1],
                                    uvd_ref.astype(int)[:, 0]] = im_ref_true

        self.estimated_reference_image = np.full(imshape, np.nan)
        self.estimated_reference_image[uvd_ref.astype(int)[:, 1],
                                       uvd_ref.astype(int)[:, 0]] = im_ref_est

        self.residual_image = np.full(imshape, np.nan)
        self.residual_image[uvd_ref.astype(int)[:, 1],
                            uvd_ref.astype(int)[:, 0]] = residual

        self.disparity_image = np.full(imshape, np.nan)
        self.disparity_image[uvd_ref.astype(int)[:, 1],
                             uvd_ref.astype(int)[:, 0]] = uvd_ref[:, 2]
