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

        for level in range(pyrlevels):
            if level == 0:
                imL = [im_left]
                imR = [im_right]
            else:
                imL.append(cv2.pyrDown(imL[-1]))
                imR.append(cv2.pyrDown(imR[-1]))

        self.image = [im.astype(float) / 255. for im in imL]

        if compute_disp:
            self.disparity = []

            for level, left, right in zip(range(pyrlevels), imL, imR):
                pyrfactor = 1. / 2.**level

                window_size = 11
                min_disp = 1
                max_disp = np.max([16, np.int(64 * pyrfactor)]) + min_disp

                stereo = cv2.StereoSGBM_create(
                    minDisparity=min_disp,
                    numDisparities=max_disp - min_disp,
                    blockSize=window_size)
                # stereo = cv2.StereoBM_create(
                # numDisparities=max_disp - min_disp, blockSize=window_size)

                disp = stereo.compute(left, right).astype(float) / 16.
                disp[disp < min_disp] = np.nan
                self.disparity.append(disp)

        if compute_jac:
            self.jacobian = []
            for level, left in zip(range(pyrlevels), imL):
                gradx = cv2.Sobel(left, -1, 1, 0)
                grady = cv2.Sobel(left, -1, 0, 1)
                self.jacobian.append(np.array([gradx.astype(float) / 255.,
                                               grady.astype(float) / 255.]))


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
        self.problem_options.max_nondecreasing_steps = 3
        self.problem_options.min_residual_decrease = 0.98

        # Number of image pyramid levels for coarse-to-fine optimization
        self.pyrlevels = 5

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
            self._compute_frame_to_frame_motion(self.keyframes[-1], trackframe)

        self.keyframes.append(trackframe)

    def _compute_frame_to_frame_motion(self, ref_frame, track_frame):
        params = {'T_1_0': SE3.identity()}

        pyrlevel_sequence = list(range(self.pyrlevels))
        pyrlevel_sequence.reverse()

        for pyrlevel in pyrlevel_sequence:
            pyrfactor = 1. / 2.**pyrlevel

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
                pyr_camera, im_ref, disp_ref, jac_ref, im_track, 1., L2Loss())

            problem = Problem(self.problem_options)
            problem.add_residual_block(residual, ['T_1_0'])
            problem.initialize_params(params)
            params = problem.solve()

        track_frame.T_c_w = params['T_1_0'] * ref_frame.T_c_w
