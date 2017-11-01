import numpy as np
import torch

import cv2


class Keyframe:
    """Keyframe base class"""

    def __init__(self, data, T_c_w=None):
        self.data = data
        """Image data (tuple or list)"""
        self.T_c_w = T_c_w
        """Keyframe pose, world-to-camera."""


class DenseKeyframe(Keyframe):
    """Dense keyframe base class"""

    def __init__(self, data, pyrimage, pyrlevels, T_c_w=None, cuda=False):
        super().__init__(data, T_c_w)

        self.pyrlevels = pyrlevels
        """Number of pyramid levels to downsample"""

        self.cuda = cuda
        """Use CUDA?"""

        self.compute_image_pyramid(pyrimage)
        """Image pyramid"""

    def compute_image_pyramid(self, pyrimage):
        """Compute an image pyramid."""
        pyrimage = torch.Tensor(pyrimage.astype(float) / 255.)

        if self.cuda:
            pyrimage = pyrimage.pin_memory().cuda(async=True)

        for pyrlevel in range(self.pyrlevels):
            if pyrlevel == 0:
                self.im_pyr = [pyrimage]
            else:
                # self.im_pyr.append(cv2.pyrDown(im_pyr[-1]))
                # need to make a copy to concatenate with other things later
                pyrimage = pyrimage[0::2, 0::2].clone()
                self.im_pyr.append(pyrimage)

    def compute_jacobian_pyramid(self):
        self.jacobian = []
        for im in self.im_pyr:
            im_np = im.cpu().numpy().astype(float)
            gradx = 0.5 * cv2.Sobel(im_np, -1, 1, 0)
            grady = 0.5 * cv2.Sobel(im_np, -1, 0, 1)
            jac = torch.Tensor(np.array([gradx, grady]))

            if self.cuda:
                jac = jac.pin_memory().cuda(async=True)

            self.jacobian.append(jac)


class DenseRGBDKeyframe(DenseKeyframe):
    """Dense RGBD keyframe"""

    def __init__(self, image, depth, pyrlevels=0, T_c_w=None, cuda=False):
        super().__init__((image, depth), image, pyrlevels, T_c_w, cuda)

    def compute_depth_pyramid(self):
        self.depth = []
        stereo = cv2.StereoBM_create()
        # stereo = cv2.StereoSGBM_create(minDisparity=0,
        #                                numDisparities=64,
        #                                blockSize=11)

        # Compute disparity at full resolution and downsample
        depth = torch.Tensor(self.data[1])
        if self.cuda:
            depth = depth.pin_memory().cuda(async=True)

        for pyrlevel in range(self.pyrlevels):
            if pyrlevel == 0:
                self.depth = [depth]
            else:
                # need to make a copy to concatenate with other things later
                depth = depth[0::2, 0::2].clone()
                self.depth.append(depth)

    def compute_pyramids(self):
        import ipdb
        ipdb.set_trace()
        self.compute_jacobian_pyramid()
        self.compute_depth_pyramid()


class DenseStereoKeyframe(DenseKeyframe):
    """Dense Stereo keyframe"""

    def __init__(self, im_left, im_right, pyrlevels=0, T_c_w=None, cuda=False):
        super().__init__((im_left, im_right), im_left, pyrlevels, T_c_w, cuda)

    @property
    def im_left(self):
        return self.data[0]

    @property
    def im_right(self):
        return self.data[1]

    def compute_disparity_pyramid(self):
        self.disparity = []
        stereo = cv2.StereoBM_create()
        # stereo = cv2.StereoSGBM_create(minDisparity=0,
        #                                numDisparities=64,
        #                                blockSize=11)

        # Compute disparity at full resolution and downsample
        disp = torch.Tensor(stereo.compute(
            self.im_left, self.im_right).astype(float) / 16.)

        if self.cuda:
            disp = disp.pin_memory().cuda(async=True)

        for pyrlevel in range(self.pyrlevels):
            if pyrlevel == 0:
                self.disparity = [disp]
            else:
                pyr_factor = 2**-pyrlevel
                # disp = cv2.pyrDown(disp) # Applies a large Gaussian blur
                # kernel!
                # need to make a copy to concatenate with other things later
                disp = disp[0::2, 0::2].clone()
                self.disparity.append(disp * pyr_factor)

    def compute_pyramids(self):
        self.compute_jacobian_pyramid()
        self.compute_disparity_pyramid()
