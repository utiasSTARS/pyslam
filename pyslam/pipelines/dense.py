import copy
import numpy as np

import cv2

from liegroups import SE3
from pyslam.problem import Options, Problem
from pyslam.sensors import StereoCamera, RGBDCamera
from pyslam.residuals import PhotometricResidualSE3
from pyslam.losses import TDistributionLoss, L2Loss, HuberLoss

from pyslam.pipelines.keyframes import DenseStereoKeyframe, DenseRGBDKeyframe


class DenseVOPipeline:
    """Base class for dense VO pipelines"""

    def __init__(self, camera, first_pose=SE3.identity()):
        self.camera = camera
        """Camera model"""
        self.first_pose = first_pose
        """First pose"""
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
        self.pyrlevel_sequence = list(range(self.pyrlevels))
        self.pyrlevel_sequence.reverse()

        self.keyframe_trans_thresh = 3.0  # meters
        """Translational distance threshold to drop new keyframes"""
        self.keyframe_rot_thresh = 0.3  # rad
        """Rotational distance threshold to drop new keyframes"""

        self.intensity_stiffness = 1. / 0.01
        """Photometric measurement stiffness"""
        self.depth_stiffness = 1. / 0.01
        """Depth or disparity measurement stiffness"""
        self.min_grad = 0.1
        """Minimum image gradient magnitude to use a given pixel"""
        self.depth_map_type = 'depth'
        """Is the depth map depth, inverse depth, disparity? ['depth','disparity'] supported"""
        self.mode = 'map'
        """Create new keyframes or localize against existing ones? ['map'|'track']"""

        self.use_motion_model_guess = True
        """Use constant motion model for initial guess."""

        # self.loss = L2Loss()
        self.loss = HuberLoss(10.0)
        # self.loss = TukeyLoss(5.0)
        # self.loss = CauchyLoss(5.0)
        # self.loss = TDistributionLoss(5.0)  # Kerl et al. ICRA 2013
        # self.loss = TDistributionLoss(3.0)
        """Loss function"""

        self._make_pyramid_cameras()

    def _make_pyramid_cameras(self):
        self.pyr_cameras = []

        for pyrlevel in self.pyrlevel_sequence:
            pyrfactor = 2**-pyrlevel

            pyr_camera = self.camera.clone()
            pyr_camera.fu *= pyrfactor
            pyr_camera.fv *= pyrfactor
            pyr_camera.cu *= pyrfactor
            pyr_camera.cv *= pyrfactor
            pyr_camera.h = int(np.ceil(pyr_camera.h * pyrfactor))
            pyr_camera.w = int(np.ceil(pyr_camera.w * pyrfactor))
            pyr_camera.compute_pixel_grid()

            self.pyr_cameras.append(pyr_camera)

    def set_mode(self, mode):
        """Set the localization mode to ['map'|'track']"""
        self.mode = mode
        if self.mode == 'track':
            self.active_keyframe_idx = 0
            self.T_c_w = []

    def track(self, trackframe, guess=None):
        """ Track an image.

            Args:
                trackframe  : frame to track
                guess       : optional initial guess of the motion
        """
        if len(self.keyframes) == 0:
            # First frame, so don't track anything yet
            trackframe.compute_pyramids()
            self.keyframes.append(trackframe)
            self.active_keyframe_idx = 0
            active_keyframe = self.keyframes[0]
        else:
            # Default behaviour for second frame and beyond
            active_keyframe = self.keyframes[self.active_keyframe_idx]

            if guess is None:
                # Default initial guess is previous pose relative to keyframe
                if len(self.T_c_w) == 0:
                    # We just started relocalizing
                    guess = SE3.identity()
                else:
                    guess = self.T_c_w[-1].dot(active_keyframe.T_c_w.inv())
                # Better initial guess is previous pose + previous motion
                if self.use_motion_model_guess and len(self.T_c_w) > 1:
                    guess = self.T_c_w[-1].dot(self.T_c_w[-2].inv().dot(guess))
            else:
                guess = guess.dot(active_keyframe.T_c_w.inv())

            # Estimate pose change from keyframe to tracking frame
            T_track_ref = self._compute_frame_to_frame_motion(
                active_keyframe, trackframe, guess)
            T_track_ref.normalize()  # Numerical instability problems otherwise
            self.T_c_w.append(T_track_ref.dot(active_keyframe.T_c_w))

            # Threshold the distance from the active keyframe to drop a new one
            se3_vec = SE3.log(T_track_ref)
            trans_dist = np.linalg.norm(se3_vec[0:3])
            rot_dist = np.linalg.norm(se3_vec[3:6])

            if trans_dist > self.keyframe_trans_thresh or \
                    rot_dist > self.keyframe_rot_thresh:
                if self.mode is 'map':
                    trackframe.T_c_w = self.T_c_w[-1]
                    trackframe.compute_pyramids()
                    self.keyframes.append(trackframe)

                    print('Dropped new keyframe. '
                          'Trans dist was {:.3f}. Rot dist was {:.3f}.'.format(
                              trans_dist, rot_dist))

                self.active_keyframe_idx += 1
                print('Active keyframe idx: {}'.format(
                    self.active_keyframe_idx))

    def _compute_frame_to_frame_motion(self, ref_frame, track_frame,
                                       guess=SE3.identity()):
        # params = {'T_1_0': guess}
        params = {'R_1_0': guess.rot, 't_1_0_1': guess.trans}

        for (pyrlevel, pyr_camera) in zip(
                self.pyrlevel_sequence, self.pyr_cameras):
            pyrfactor = 2**-pyrlevel

            im_jacobian = ref_frame.jacobian[pyrlevel]
            # ESM
            # im_jacobian = 0.5 * (ref_frame.jacobian[pyrlevel] +
            #                      track_frame.jacobian[pyrlevel])

            if self.depth_map_type is 'disparity':
                depth_ref = ref_frame.disparity[pyrlevel]
                # Disparity is in pixels, so we need to scale it according to the pyramid level
                depth_stiffness = self.depth_stiffness / pyrfactor
            else:
                depth_ref = ref_frame.depth[pyrlevel]
                depth_stiffness = self.depth_stiffness

            residual = PhotometricResidualSE3(pyr_camera,
                                              ref_frame.im_pyr[pyrlevel],
                                              depth_ref,
                                              track_frame.im_pyr[pyrlevel],
                                              im_jacobian,
                                              self.intensity_stiffness,
                                              depth_stiffness,
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


class DenseStereoPipeline(DenseVOPipeline):
    """Dense stereo VO pipeine"""

    def __init__(self, camera, first_pose=SE3.identity()):
        super().__init__(camera, first_pose)
        self.depth_map_type = 'disparity'
        self.depth_stiffness = 1 / 0.5

    def track(self, im_left, im_right, guess=None):
        if len(self.keyframes) == 0:
            # First frame, so create first keyframe with given initial pose
            trackframe = DenseStereoKeyframe(im_left, im_right, self.pyrlevels,
                                             self.T_c_w[0])
        else:
            # Default behaviour for second frame and beyond
            trackframe = DenseStereoKeyframe(im_left, im_right, self.pyrlevels)

        super().track(trackframe, guess)


class DenseRGBDPipeline(DenseVOPipeline):
    """Dense RGBD VO pipeline"""

    def __init__(self, camera, first_pose=SE3.identity()):
        super().__init__(camera, first_pose)
        self.depth_map_type = 'depth'
        self.depth_stiffness = 1 / 0.01

    def track(self, image, depth, guess=None):
        if len(self.keyframes) == 0:
            # First frame, so create first keyframe with given initial pose
            trackframe = DenseRGBDKeyframe(image, depth, self.pyrlevels,
                                           self.T_c_w[0])
        else:
            # Default behaviour for second frame and beyond
            trackframe = DenseRGBDKeyframe(image, depth, self.pyrlevels)

        super().track(trackframe, guess)
