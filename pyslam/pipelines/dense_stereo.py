import numpy as np
import copy

import cv2

from liegroups import SE3
from pyslam.problem import Options, Problem
from pyslam.sensors import StereoCamera
from pyslam.residuals import PhotometricResidual
from pyslam.losses import HuberLoss, L2Loss
from pyslam.utils import invsqrt


class DenseKeyframe:
    """Dense keyframe"""

    def __init__(self, im_left, im_right, pyrlevels=0,
                 T_c_w=SE3.identity(), T_c_w_covar=np.identity(6),
                 compute_disp=False, compute_jac=False):
        self.pyrlevels = pyrlevels
        self.T_c_w = T_c_w
        self.T_c_w_covar = T_c_w_covar

        for pyrlevel in range(pyrlevels):
            if pyrlevel == 0:
                pyr_left = [im_left]
                pyr_right = [im_right]
            else:
                pyr_left.append(cv2.pyrDown(pyr_left[-1]))
                pyr_right.append(cv2.pyrDown(pyr_right[-1]))

        self.image = [im.astype(float) / 255. for im in pyr_left]

        if compute_disp:
            self.disparity = []
            stereo = cv2.StereoBM_create()
            # stereo = cv2.StereoSGBM_create(minDisparity=0,
            #                                numDisparities=64,
            #                                blockSize=11)

            # Method 1: Compute disparity at full resolution and downsample
            disp = stereo.compute(im_left, im_right).astype(float) / 16.
            disp[disp < 0] = np.nan

            for pyrlevel in range(pyrlevels):
                if pyrlevel == 0:
                    self.disparity = [disp]
                else:
                    pyrfactor = 2**-pyrlevel
                    # disp = cv2.pyrDown(disp) # Applies a large Gaussian blur
                    # kernel!
                    disp = disp[0::2, 0::2]
                    self.disparity.append(disp * pyrfactor)

            # Method 2: Compute disparity on downsampled images
            # for level, imL, imR in zip(range(pyrlevels), pyr_left, pyr_right):
            #     disp = stereo.compute(imL, imR).astype(float) / 16.
            #     disp[disp <= 0] = np.nan
            #     self.disparity.append(disp)

        if compute_jac:
            self.jacobian = []

            for im in self.image:
                gradx = 0.5 * cv2.Sobel(im, -1, 1, 0)
                grady = 0.5 * cv2.Sobel(im, -1, 0, 1)
                self.jacobian.append(np.array([gradx, grady]))


class DenseStereoPipeline:
    """Dense stereo VO pipeline"""

    def __init__(self, camera, first_pose=SE3.identity(),
                 first_pose_covar=1e-6 * np.identity(6)):
        self.camera = camera
        """Camera model"""
        self.first_pose = first_pose
        """First pose (transformation from world frame to camera frame)"""
        self.first_pose_covar = first_pose_covar
        """Covariance matrix of first pose"""
        self.keyframes = []
        """List of keyframes"""
        self.problem_options = Options()
        """Optimizer parameters"""

        # Default optimizer parameters
        self.problem_options.allow_nondecreasing_steps = True
        self.problem_options.max_nondecreasing_steps = 5
        self.problem_options.min_cost_decrease = 0.99
        self.problem_options.max_iters = 40

        # Number of image pyramid levels for coarse-to-fine optimization
        self.pyrlevels = 6

    def track(self, im_left, im_right):
        if len(self.keyframes) is 0:
            # First frame, so don't track anything yet
            trackframe = DenseKeyframe(im_left, im_right, self.pyrlevels,
                                       self.first_pose, self.first_pose_covar,
                                       compute_disp=True, compute_jac=True)
        else:
            # Default behaviour for second frame and beyond
            trackframe = DenseKeyframe(im_left, im_right, self.pyrlevels,
                                       compute_disp=True, compute_jac=True)
            T_1_0 = self._compute_frame_to_frame_motion(
                self.keyframes[-1], trackframe)
            trackframe.T_c_w = T_1_0 * self.keyframes[-1].T_c_w

        self.keyframes.append(trackframe)

    def _compute_frame_to_frame_motion(self, ref_frame, track_frame):
        params = {'T_1_0': SE3.identity()}

        pyrlevel_sequence = list(range(self.pyrlevels))
        pyrlevel_sequence.reverse()

        stiffness = 1. / 0.05
        # loss = L2Loss()
        loss = HuberLoss(1.345)

        for pyrlevel in pyrlevel_sequence[:-1]:
            pyrfactor = 2**-pyrlevel

            im_ref = ref_frame.image[pyrlevel]
            disp_ref = ref_frame.disparity[pyrlevel]
            jac_ref = ref_frame.jacobian[pyrlevel]
            im_track = track_frame.image[pyrlevel]

            pyr_camera = copy.deepcopy(self.camera)
            pyr_camera.fu *= pyrfactor
            pyr_camera.fv *= pyrfactor
            pyr_camera.cu *= pyrfactor
            pyr_camera.cv *= pyrfactor
            pyr_camera.w = im_ref.shape[1]
            pyr_camera.h = im_ref.shape[0]

            residual = PhotometricResidual(
                pyr_camera, im_ref, disp_ref, jac_ref, im_track,
                stiffness, loss)

            # import ipdb
            # ipdb.set_trace()

            problem = Problem(self.problem_options)
            problem.add_residual_block(residual, ['T_1_0'])
            problem.initialize_params(params)
            params = problem.solve()

        return params['T_1_0']
