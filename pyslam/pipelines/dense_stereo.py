import numpy as np
import copy

import cv2

from liegroups import SE3
from pyslam.problem import Options, Problem
from pyslam.sensors import StereoCamera
from pyslam.residuals import PhotometricResidual
from pyslam.losses import HuberLoss, L2Loss, TukeyLoss
from pyslam.utils import invsqrt


class DenseKeyframe:
    """Dense keyframe"""

    def __init__(self, im_left, im_right, pyrlevels=0, T_c_w=SE3.identity()):
        self.im_left = im_left
        self.im_right = im_right
        self.pyrlevels = pyrlevels
        self.T_c_w = T_c_w

        # Compute image pyramid
        for pyrlevel in range(self.pyrlevels):
            if pyrlevel == 0:
                pyr_left = [im_left]
                # pyr_right = [im_right]
            else:
                pyr_left.append(cv2.pyrDown(pyr_left[-1]))
                # pyr_right.append(cv2.pyrDown(pyr_right[-1]))

        self.im_pyr = [im.astype(float) / 255. for im in pyr_left]

    def compute_jacobian(self):
        self.jacobian = []
        for im in self.im_pyr:
            gradx = 0.5 * cv2.Sobel(im, -1, 1, 0)
            grady = 0.5 * cv2.Sobel(im, -1, 0, 1)
            self.jacobian.append(np.array([gradx, grady]))

    def compute_disparity(self):
        self.disparity = []
        stereo = cv2.StereoBM_create()
        # stereo = cv2.StereoSGBM_create(minDisparity=0,
        #                                numDisparities=64,
        #                                blockSize=11)

        # Compute disparity at full resolution and downsample
        disp = stereo.compute(self.im_left, self.im_right).astype(float) / 16.
        disp[disp < 0] = np.nan

        for pyrlevel in range(self.pyrlevels):
            if pyrlevel == 0:
                self.disparity = [disp]
            else:
                pyrfactor = 2**-pyrlevel
                # disp = cv2.pyrDown(disp) # Applies a large Gaussian blur
                # kernel!
                disp = disp[0::2, 0::2]
                self.disparity.append(disp * pyrfactor)

    def compute_jacobian_and_disparity(self):
        self.compute_jacobian()
        self.compute_disparity()


class DenseStereoPipeline:
    """Dense stereo VO pipeline"""

    def __init__(self, camera, first_pose=SE3.identity()):
        self.camera = camera
        """Camera model"""
        self.keyframes = []
        """List of keyframes"""
        self.T_c_w = [first_pose]
        """List of camera poses"""
        self.problem_options = Options()
        """Optimizer parameters"""

        # Default optimizer parameters
        self.problem_options.allow_nondecreasing_steps = True
        self.problem_options.max_nondecreasing_steps = 5
        self.problem_options.min_cost_decrease = 0.99
        self.problem_options.max_iters = 30

        # Number of image pyramid levels for coarse-to-fine optimization
        self.pyrlevels = 6

    def track(self, im_left, im_right):
        # import ipdb
        # ipdb.set_trace()
        if len(self.keyframes) == 0:
            # First frame, so don't track anything yet
            trackframe = DenseKeyframe(im_left, im_right, self.pyrlevels,
                                       self.T_c_w[0])
            trackframe.compute_jacobian_and_disparity()
            self.keyframes.append(trackframe)
        else:
            # Default behaviour for second frame and beyond
            trackframe = DenseKeyframe(im_left, im_right, self.pyrlevels)

            guess = self.T_c_w[-1] * self.keyframes[-1].T_c_w.inv()
            # print('guess = \n{}'.format(SE3.log(guess)))
            T_track_ref = self._compute_frame_to_frame_motion(
                self.keyframes[-1], trackframe, guess)
            # print('est = \n{}'.format(SE3.log(T_track_ref)))

            self.T_c_w.append(T_track_ref * self.keyframes[-1].T_c_w)

            # Threshold the distance from the active keyframe to drop a new one
            se3_vec = SE3.log(T_track_ref)
            trans_dist = np.linalg.norm(se3_vec[0:3])
            rot_dist = np.linalg.norm(se3_vec[3:6])
            print('trans_dist = {}, rot_dist = {}'.format(trans_dist, rot_dist))
            if trans_dist > 2 or rot_dist > 0.2:
                trackframe.T_c_w = self.T_c_w[-1]
                trackframe.T_c_w.normalize()  # Numerical instability problems otherwise
                trackframe.compute_jacobian_and_disparity()
                self.keyframes.append(trackframe)
                print('Dropped new keyframe. Now have {}.'.format(
                    len(self.keyframes)))

    def _compute_frame_to_frame_motion(self, ref_frame, track_frame,
                                       guess=SE3.identity()):
        params = {'T_1_0': guess}

        pyrlevel_sequence = list(range(self.pyrlevels))
        pyrlevel_sequence.reverse()

        stiffness = 1. / 0.05
        # loss = L2Loss()
        # loss = HuberLoss(1.345)
        loss = TukeyLoss(1.345)

        for pyrlevel in pyrlevel_sequence[:-1]:
            pyrfactor = 2**-pyrlevel

            im_ref = ref_frame.im_pyr[pyrlevel]
            disp_ref = ref_frame.disparity[pyrlevel]
            jac_ref = ref_frame.jacobian[pyrlevel]
            im_track = track_frame.im_pyr[pyrlevel]

            pyr_camera = copy.deepcopy(self.camera)
            pyr_camera.fu *= pyrfactor
            pyr_camera.fv *= pyrfactor
            pyr_camera.cu *= pyrfactor
            pyr_camera.cv *= pyrfactor
            pyr_camera.w = im_ref.shape[1]
            pyr_camera.h = im_ref.shape[0]

            residual = PhotometricResidual(
                pyr_camera, im_ref, disp_ref, jac_ref, im_track,
                stiffness)

            problem = Problem(self.problem_options)
            problem.add_residual_block(residual, ['T_1_0'], loss=loss)
            problem.initialize_params(params)
            params = problem.solve()

        return params['T_1_0']

    def _optimize_keyframe_graph(self):
        pass
