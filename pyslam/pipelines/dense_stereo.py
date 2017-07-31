import copy
import numpy as np

import cv2

from liegroups import SE3
from pyslam.problem import Options, Problem
from pyslam.sensors import StereoCamera
from pyslam.residuals import PhotometricResidualSE3
from pyslam.losses import TDistributionLoss, L2Loss, HuberLoss


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
        self.motion_options = Options()
        """Optimizer parameters for motion estimation"""

        # Default optimizer parameters for motion estimation
        self.motion_options.allow_nondecreasing_steps = True
        self.motion_options.max_nondecreasing_steps = 5
        self.motion_options.min_cost_decrease = 0.99
        self.motion_options.max_iters = 30
        self.motion_options.num_threads = 1
        self.motion_options.linesearch_max_iters = 0

        self.pyrlevels = 4
        """Number of image pyramid levels for coarse-to-fine optimization"""
        self.pyrlevel_sequence = list(range(self.pyrlevels))[1:]
        self.pyrlevel_sequence.reverse()

        self.keyframe_trans_thresh = 3.0  # meters
        """Translational distance threshold to drop new keyframes"""
        self.keyframe_rot_thresh = 0.3  # rad
        """Rotational distance threshold to drop new keyframes"""

        self.intensity_stiffness = 1. / 0.01
        """Photometric measurement stiffness"""
        self.disparity_stiffness = 1. / 0.5
        """Disparity measurement stiffness"""
        self.min_grad = 0.1
        """Minimum image gradient magnitude to use a given pixel"""

        # self.loss = L2Loss()
        self.loss = HuberLoss(10.0)
        # self.loss = TukeyLoss(5.0)
        # self.loss = CauchyLoss(5.0)
        # self.loss = TDistributionLoss(5.0)  # Kerl et al. ICRA 2013
        # self.loss = TDistributionLoss(3.0)
        """Loss function"""

    def track(self, im_left, im_right, guess=None):
        if len(self.keyframes) == 0:
            # First frame, so don't track anything yet
            trackframe = DenseKeyframe(im_left, im_right, self.pyrlevels,
                                       self.T_c_w[0])
            trackframe.compute_jacobian_and_disparity()
            self.keyframes.append(trackframe)

        else:
            # Default behaviour for second frame and beyond
            trackframe = DenseKeyframe(im_left, im_right, self.pyrlevels)
            # trackframe.compute_jacobian()

            if guess is None:
                # Default initial guess is previous pose relative to keyframe
                guess = self.T_c_w[-1].dot(self.keyframes[-1].T_c_w.inv())
                # Better initial guess is previous pose + previous motion
                if len(self.T_c_w) > 1:
                    guess = self.T_c_w[-1].dot(self.T_c_w[-2].inv().dot(guess))
            else:
                guess = guess.dot(self.keyframes[-1].T_c_w.inv())

            # Estimate pose change from keyframe to tracking frame
            T_track_ref = self._compute_frame_to_frame_motion(
                self.keyframes[-1], trackframe, guess)
            T_track_ref.normalize()  # Numerical instability problems otherwise
            self.T_c_w.append(T_track_ref.dot(self.keyframes[-1].T_c_w))

            # Threshold the distance from the active keyframe to drop a new one
            se3_vec = SE3.log(T_track_ref)
            trans_dist = np.linalg.norm(se3_vec[0:3])
            rot_dist = np.linalg.norm(se3_vec[3:6])

            if trans_dist > self.keyframe_trans_thresh or \
                    rot_dist > self.keyframe_rot_thresh:
                trackframe.T_c_w = self.T_c_w[-1]
                trackframe.compute_jacobian_and_disparity()
                self.keyframes.append(trackframe)

                print('Dropped new keyframe. '
                      'Trans dist was {:.3f}. Rot dist was {:.3f}. Now have {}.'.format(trans_dist, rot_dist, len(self.keyframes)))

    def _compute_frame_to_frame_motion(self, ref_frame, track_frame,
                                       guess=SE3.identity()):
        # params = {'T_1_0': guess}
        params = {'R_1_0': guess.rot, 't_1_0_1': guess.trans}

        for pyrlevel in self.pyrlevel_sequence:
            pyrfactor = 2**-pyrlevel

            pyr_camera = copy.deepcopy(self.camera)
            pyr_camera.fu *= pyrfactor
            pyr_camera.fv *= pyrfactor
            pyr_camera.cu *= pyrfactor
            pyr_camera.cv *= pyrfactor
            pyr_camera.h, pyr_camera.w = ref_frame.im_pyr[pyrlevel].shape

            im_jacobian = ref_frame.jacobian[pyrlevel]
            # ESM
            # im_jacobian = 0.5 * (ref_frame.jacobian[pyrlevel] +
            #                      track_frame.jacobian[pyrlevel])

            residual = PhotometricResidualSE3(pyr_camera,
                                              ref_frame.im_pyr[pyrlevel],
                                              ref_frame.disparity[pyrlevel],
                                              track_frame.im_pyr[pyrlevel],
                                              im_jacobian,
                                              self.intensity_stiffness,
                                              self.disparity_stiffness / pyrfactor,
                                              self.min_grad)

            problem = Problem(self.motion_options)
            # problem.add_residual_block(residual, ['T_1_0'], loss=self.loss)
            problem.add_residual_block(
                residual, ['R_1_0', 't_1_0_1'], loss=self.loss)
            problem.initialize_params(params)

            if pyrlevel > 2:
                problem.set_parameters_constant('t_1_0_1')
            # else:
            # problem.set_parameters_constant('R_1_0')

            params = problem.solve()
            # print(problem.summary(format='brief'))

            # # DEBUG: Store residuals for later
            # try:
            #     self.residuals = np.hstack(
            #         [self.residuals, residual.evaluate([guess])])
            # except AttributeError:
            #     self.residuals = residual.evaluate([guess])

        # return params['T_1_0']
        return SE3(params['R_1_0'], params['t_1_0_1'])
