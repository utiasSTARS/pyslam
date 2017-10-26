import numpy as np
import torch
import time

from liegroups.torch import SO3, SE3
from pyslam.utils import bilinear_interpolate


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

    def __init__(self, camera, im_ref, depth_ref, im_track, im_jac,
                 intensity_stiffness, depth_stiffness, min_grad=0.):
        """Depth image and stiffness parameter can also be disparity."""
        self.use_cuda = im_ref.is_cuda

        self.camera = camera
        self.im_ref = torch.FloatTensor(im_ref).view(-1)
        self.im_ref.pin_memory()

        u_range = range(0, self.camera.w)
        v_range = range(0, self.camera.h)
        u_coords, v_coords = np.meshgrid(u_range, v_range, indexing='xy')

        u_coords = torch.FloatTensor(u_coords.astype(np.float)).view(-1, 1)
        v_coords = torch.FloatTensor(v_coords.astype(np.float)).view(-1, 1)
        d_coords = torch.FloatTensor(depth_ref).view(-1, 1)

        self.uvd_ref = torch.cat(
            [u_coords, v_coords, d_coords], dim=1).pin_memory()

        self.im_jac = im_jac.view(2, -1).transpose_(0, 1)

        self.im_track = im_track

        self.intensity_stiffness = intensity_stiffness
        self.depth_stiffness = depth_stiffness
        self.intensity_covar = intensity_stiffness ** -2
        self.depth_covar = depth_stiffness ** -2

        self.min_grad = min_grad

        # Filter out invalid pixels (NaN or negative depth)
        valid_pixels = self.camera.is_valid_measurement(
            self.uvd_ref).nonzero().squeeze_()
        self.uvd_ref = self.uvd_ref[valid_pixels, :]
        self.im_ref = self.im_ref[valid_pixels]
        self.im_jac = self.im_jac[valid_pixels, :]

        # Filter out pixels with weak gradients
        grad_ref = self.im_jac.norm(p=2, dim=1)
        strong_pixels = (grad_ref >= self.min_grad).nonzero().squeeze_()
        self.uvd_ref = self.uvd_ref[strong_pixels, :]
        self.im_ref = self.im_ref[strong_pixels]
        self.im_jac = self.im_jac[strong_pixels, :]

        # Precompute triangulated 3D points
        self.pt_ref, self.triang_jac = self.camera.triangulate(
            self.uvd_ref, compute_jacobians=True)
        self.pt_ref = self.pt_ref.pin_memory()
        self.triang_jac = self.triang_jac.pin_memory()

        # Move to GPU
        if self.use_cuda:
            self.im_ref = self.im_ref.cuda(async=True)
            self.im_track = self.im_track.cuda(async=True)
            self.uvd_ref = self.uvd_ref.cuda(async=True)
            self.im_jac = self.im_jac.cuda(async=True)
            self.pt_ref = self.pt_ref.cuda(async=True)
            self.triang_jac = self.triang_jac.cuda(async=True)

    def evaluate(self, params, compute_jacobians=None):
        if len(params) == 1:
            T_track_ref = SE3.from_numpy(params[0], pin_memory=True)
        elif len(params) == 2:
            T_track_ref = SE3(SO3.from_numpy(params[0], pin_memory=True),
                              torch.Tensor(params[1]).pin_memory())
        else:
            raise ValueError(
                'In PhotometricResidual.evaluate() params must have length 1 or 2')

        if self.use_cuda:
            T_track_ref = T_track_ref.cuda(async=True)

        # Reproject reference image pixels into tracking image to predict the
        # reference image based on the tracking image
        pt_track = T_track_ref.dot(self.pt_ref)
        uvd_track, project_jac = self.camera.project(
            pt_track, compute_jacobians=True)

        # Filter out bad measurements
        # where returns a tuple
        valid_pixels = self.camera.is_valid_measurement(
            uvd_track).nonzero().squeeze_()

        pt_track = pt_track[valid_pixels, :]
        uvd_track = uvd_track[valid_pixels, :]
        im_jac = self.im_jac[valid_pixels, :].unsqueeze_(dim=1)
        project_jac = project_jac[valid_pixels, :, :]
        triang_jac = self.triang_jac[valid_pixels, :, :]

        # The residual is the intensity difference between the estimated
        # reference image pixels and the true reference image pixels
        im_ref_est = bilinear_interpolate(self.im_track,
                                          uvd_track[:, 0],
                                          uvd_track[:, 1])
        residual = im_ref_est - self.im_ref[valid_pixels]

        # We need the jacobian of the residual w.r.t. the depth
        # to compute a reasonable stiffness paramater
        im_proj_jac = im_jac.bmm(project_jac[:, 0:2, :])  # Nx1x3
        im_depth_jac = im_proj_jac.bmm(
            T_track_ref.rot.as_matrix().unsqueeze(dim=0).expand(
                im_proj_jac.shape[0], 3, 3)).bmm(
            triang_jac[:, :, 2:3])  # Nx1x1

        # Compute the overall stiffness
        # \sigma^2 = \sigma^2_I + J_d \sigma^2_d J_d^T
        stiffness = 1. / (self.intensity_covar +
                          self.depth_covar * im_depth_jac**2).sqrt_().squeeze_()
        # stiffness = self.intensity_stiffness
        residual = stiffness * residual
        residual = residual.cpu().numpy().astype(float)

        # DEBUG: Rebuild residual and depth images
        # self._rebuild_images(residual,
        #                      im_ref_est.cpu().numpy().astype(float),
        #                      self.im_ref.cpu().numpy().astype(float),
        #                      valid_pixels.cpu().numpy().astype(int))
        # import ipdb
        # ipdb.set_trace()

        # Jacobian time!
        if compute_jacobians:
            if any(compute_jacobians):
                jac = im_proj_jac.bmm(SE3.odot(pt_track)).squeeze_()
                jac = stiffness.unsqueeze(dim=1).expand_as(jac) * jac
                jac = jac.cpu().numpy().astype(float)

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
        residual and depth images as a sanity check"""
        uvd_ref = self.uvd_ref.cpu().numpy().astype(float)[valid_pixels, :]
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

        self.depth_image = np.full(imshape, np.nan)
        self.depth_image[
            uvd_ref.astype(int)[:, 1],
            uvd_ref.astype(int)[:, 0]] = uvd_ref[:, 2]
