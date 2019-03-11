import numpy as np

import cv2

from liegroups import SE3


class Keyframe:
    """Keyframe base class"""

    def __init__(self, data, T_c_w=SE3.identity()):
        self.data = data
        """Image data (tuple or list)"""
        self.T_c_w = T_c_w
        """Keyframe pose, world-to-camera."""


class DenseKeyframe(Keyframe):
    """Dense keyframe base class"""

    def __init__(self, data, pyrimage, pyrlevels, T_c_w=SE3.identity()):
        super().__init__(data, T_c_w)

        self.pyrlevels = pyrlevels
        """Number of pyramid levels to downsample"""

        self.compute_image_pyramid(pyrimage)
        """Image pyramid"""

    def compute_image_pyramid(self, pyrimage):
        """Compute an image pyramid."""

        for pyrlevel in range(self.pyrlevels):
            if pyrlevel == 0:
                im_pyr = [pyrimage]
            else:
                im_pyr.append(cv2.pyrDown(im_pyr[-1]))

        self.im_pyr = [im.astype(float) / 255. for im in im_pyr]

    def compute_jacobian_pyramid(self):
        self.jacobian = []
        for im in self.im_pyr:
            gradx = 0.5 * cv2.Sobel(im, -1, 1, 0)
            grady = 0.5 * cv2.Sobel(im, -1, 0, 1)
            self.jacobian.append(np.array([gradx, grady]))


class DenseRGBDKeyframe(DenseKeyframe):
    """Dense RGBD keyframe"""

    def __init__(self, image, depth, pyrlevels=0, T_c_w=SE3.identity()):
        super().__init__((image, depth), image, pyrlevels, T_c_w)

    def compute_depth_pyramid(self):
        self.depth = []
        stereo = cv2.StereoBM_create()
        # stereo = cv2.StereoSGBM_create(minDisparity=0,
        #                                numDisparities=64,
        #                                blockSize=11)

        # Compute disparity at full resolution and downsample
        depth = self.data[1]

        for pyrlevel in range(self.pyrlevels):
            if pyrlevel == 0:
                self.depth = [depth]
            else:
                pyr_factor = 2**-pyrlevel
                depth = depth[0::2, 0::2]
                self.depth.append(depth)

    def compute_pyramids(self):
        self.compute_jacobian_pyramid()
        self.compute_depth_pyramid()


class DenseStereoKeyframe(DenseKeyframe):
    """Dense Stereo keyframe"""

    def __init__(self, im_left, im_right, pyrlevels=0, T_c_w=SE3.identity()):
        super().__init__((im_left, im_right), im_left, pyrlevels, T_c_w)

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
        disp = stereo.compute(self.im_left, self.im_right).astype(float) / 16.

        for pyrlevel in range(self.pyrlevels):
            if pyrlevel == 0:
                self.disparity = [disp]
            else:
                pyr_factor = 2**-pyrlevel
                # disp = cv2.pyrDown(disp) # Applies a large Gaussian blur
                # kernel!
                disp = disp[0::2, 0::2]
                self.disparity.append(disp * pyr_factor)

    def compute_pyramids(self):
        self.compute_jacobian_pyramid()
        self.compute_disparity_pyramid()


class SparseStereoKeyframe(Keyframe):
    """Sparse Stereo keyframe"""

    def __init__(self, im_left, im_right, T_c_w=SE3.identity()):
        super().__init__((im_left, im_right), T_c_w)

    @property
    def im_left(self):
        return self.data[0]

    @property
    def im_right(self):
        return self.data[1]


class SparseRGBDKeyframe(Keyframe):
    """Sparse RGB-D keyframe"""

    def __init__(self, image, depth,  T_c_w=SE3.identity()):
        super().__init__((image, depth), T_c_w)

    @property
    def image(self):
        return self.data[0]

    @property
    def depth(self):
        return self.data[1]
