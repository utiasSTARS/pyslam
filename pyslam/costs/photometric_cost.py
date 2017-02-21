import numpy as np
import time

from liegroups import SE3

from pyslam.utils import bilinear_interpolate, stackmul
from numba import guvectorize, float64


@guvectorize([(float64[:, :], float64[:], float64[:])], '(n,n),(m)->(m)', nopython=True, target='parallel')
def parallel_SE3_transform(T, p, out):
    out[0] = T[0, 0] * p[0] + T[0, 1] * p[1] + T[0, 2] * p[2] + T[0, 3]
    out[1] = T[1, 0] * p[0] + T[1, 1] * p[1] + T[1, 2] * p[2] + T[1, 3]
    out[2] = T[2, 0] * p[0] + T[2, 1] * p[1] + T[2, 2] * p[2] + T[2, 3]


@guvectorize([(float64[:], float64[:], float64[:, :])], '(n),(m)->(n,m)', nopython=True, target='parallel')
def parallel_SE3_odot(p, junk, out):
    out[0, 0] = 1.
    out[0, 1] = 0.
    out[0, 2] = 0.
    out[0, 3] = 0.
    out[0, 4] = p[2]
    out[0, 5] = -p[1]

    out[1, 0] = 0.
    out[1, 1] = 1.
    out[1, 2] = 0.
    out[1, 3] = -p[2]
    out[1, 4] = 0.
    out[1, 5] = p[0]

    out[2, 0] = 0.
    out[2, 1] = 0.
    out[2, 2] = 1.
    out[2, 3] = p[1]
    out[2, 4] = -p[0]
    out[2, 5] = 0.


@guvectorize([(float64[:], float64, float64, float64, float64, float64, float64[:])], '(n),(),(),(),(),()->(n)', nopython=True, target='parallel')
def parallel_stereo_project(p, fu, fv, cu, cv, b, out):
    one_over_z = 1. / p[2]
    out[0] = fu * p[0] * one_over_z + cu
    out[1] = fv * p[1] * one_over_z + cv
    out[2] = fu * b * one_over_z


@guvectorize([(float64[:], float64, float64, float64, float64, float64, float64[:, :])], '(n),(),(),(),(),()->(n,n)', nopython=True, target='parallel')
def parallel_stereo_project_jacobian(p, fu, fv, cu, cv, b, out):
    one_over_z = 1. / p[2]
    one_over_z2 = one_over_z * one_over_z

    # d(u) / d(p)
    out[0, 0] = fu * one_over_z
    out[0, 1] = 0.
    out[0, 2] = -fu * p[0] * one_over_z2

    # d(v) / d(p)
    out[1, 0] = 0.
    out[1, 1] = fv * one_over_z
    out[1, 2] = -fv * p[1] * one_over_z2

    # d(d) / d(p)
    out[2, 0] = 0.
    out[2, 1] = 0.
    out[2, 2] = -fu * b * one_over_z2


class PhotometricCost:
    """Photometric cost for greyscale images.
    Uses the pre-computed reference image jacobian as an approximation to the
    tracking image jacobian under the assumption that the camera motion is small."""

    def __init__(self, camera, im_ref, disp_ref, jac_ref, im_track, stiffness):
        self.camera = camera
        self.stiffness = stiffness
        self.im_track = im_track

        self.im_ref = im_ref.flatten()

        u, v = np.meshgrid(list(range(0, camera.w)),
                           list(range(0, camera.h)), indexing='xy')
        self.uvd_ref = np.array(
            [u.flatten(), v.flatten(), disp_ref.flatten()]).T

        self.jac_ref = np.array([jac_ref[0, :, :].flatten(),
                                 jac_ref[1, :, :].flatten()]).T

        # Filter out invalid pixels (NaN or negative disparity)
        valid_pixels = self.camera.is_valid_measurement(self.uvd_ref)
        self.uvd_ref = self.uvd_ref[valid_pixels, :]
        self.im_ref = self.im_ref[valid_pixels]
        self.jac_ref = self.jac_ref[valid_pixels, :]

        # Filter out pixels with weak gradients
        grad_ref = np.sum(self.jac_ref**2, axis=1)
        strong_pixels = grad_ref >= 0.5**2
        self.uvd_ref = self.uvd_ref[strong_pixels, :]
        self.im_ref = self.im_ref[strong_pixels]
        self.jac_ref = self.jac_ref[strong_pixels, :]

        # Precompute triangulated 3D points
        self.pt_ref = self.camera.triangulate(np.array(self.uvd_ref))

    def evaluate(self, params, compute_jacobians=None):
        T_track_ref = params[0]

        # Reproject reference image pixels into tracking image to predict the
        # reference image based on the tracking image
        im_ref_true = self.im_ref

        # pt_track = T_track_ref * self.pt_ref
        pt_track = np.empty(self.pt_ref.shape)
        parallel_SE3_transform(T_track_ref.as_matrix(), self.pt_ref, pt_track)

        uvd_track = np.empty(pt_track.shape)
        parallel_stereo_project(pt_track, self.camera.fu, self.camera.fv,
                                self.camera.cu, self.camera.cv, self.camera.b, uvd_track)
        if compute_jacobians:
            im_jac = self.jac_ref
            # uvd_track, project_jac = self.camera.project(
            #     pt_track, compute_jacobians=True)
            project_jac = np.empty(
                [pt_track.shape[0], pt_track.shape[1], pt_track.shape[1]])
            parallel_stereo_project_jacobian(pt_track,
                                             self.camera.fu, self.camera.fv,
                                             self.camera.cu, self.camera.cv,
                                             self.camera.b, project_jac)
        # else:
        #     uvd_track = self.camera.project(pt_track)

        # Filter out bad measurements (out of bounds coordinates, nonpositive
        # disparity)
        valid_track = self.camera.is_valid_measurement(uvd_track)
        uvd_track = uvd_track[valid_track, :]
        pt_track = pt_track[valid_track, :]
        im_ref_true = im_ref_true[valid_track]
        if compute_jacobians:
            im_jac = im_jac[valid_track, :]
            project_jac = project_jac[valid_track, :, :]

        # The residual is the intensity difference between the estimated
        # reference image pixels and the true reference image pixels
        im_ref_est = im_ref_true.shape
        bilinear_interpolate(self.im_track, uvd_track[
                             :, 0], uvd_track[:, 1], im_ref_est)
        # im_ref_est = bilinear_interpolate(
        #     self.im_track, uvd_track[:, 0], uvd_track[:, 1])
        residual = self.stiffness * (im_ref_est - im_ref_true)

        # DEBUG: Rebuild residual and disparity images
        # self._rebuild_images(residual, im_ref_est, im_ref_true, valid_track)

        # Jacobian time!
        if compute_jacobians:
            jacobians = [None for _ in enumerate(params)]

            if compute_jacobians[0]:
                # jacobians[0] = self.stiffness * np.einsum('...i,...ij,...jk->...k',
                # im_jac, project_jac[:, 0:2, :], SE3.odot(pt_track))
                jacobians[0] = np.empty([im_jac.shape[0], 1, 6])
                temp = np.empty([im_jac.shape[0], 1, 3])
                im_jac = np.expand_dims(im_jac, axis=1)

                # start = time.perf_counter()
                # np.matmul(im_jac, project_jac[:, 0:2, :], out=temp)
                stackmul(im_jac, project_jac[:, 0:2, :], temp)
                # end = time.perf_counter()
                # print('mult1 {:5} sec'.format(end - start))

                # start = time.perf_counter()
                # odot_pt_track = SE3.odot(pt_track)
                odot_pt_track = np.empty([im_jac.shape[0], 3, 6])
                parallel_SE3_odot(pt_track, np.empty(6), odot_pt_track)
                # np.matmul(temp, odot_pt_track, out=jacobians[0])
                stackmul(temp, odot_pt_track, jacobians[0])
                # end = time.perf_counter()
                # print('mult2 {:5} sec'.format(end - start))

                jacobians[0] = np.squeeze(jacobians[0])

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
