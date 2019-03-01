import copy
import numpy as np

import cv2

from liegroups import SE3
import viso2

from pyslam.problem import Options, Problem
from pyslam.sensors import StereoCamera, RGBDCamera
from pyslam.residuals import ReprojectionMotionOnlyBatchResidual
from pyslam.losses import L2Loss

from pyslam.pipelines.keyframes import SparseStereoKeyframe, SparseRGBDKeyframe
from pyslam.pipelines.ransac import FrameToFrameRANSAC


class SparseVOPipeline:
    """Base class for sparse VO pipelines"""

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

        self.keyframe_trans_thresh = 3.0  # meters
        """Translational distance threshold to drop new keyframes"""
        self.keyframe_rot_thresh = 0.3  # rad
        """Rotational distance threshold to drop new keyframes"""

        self.matcher_params = viso2.Matcher_parameters()
        """Parameters for libviso2 matcher"""
        self.matcher = viso2.Matcher(self.matcher_params)
        """libviso2 matcher"""
        self.matcher_mode = 0
        """libviso2 matching mode 0=flow 1=stereo 2=quad"""

        self.ransac = FrameToFrameRANSAC(self.camera)
        """RANSAC outlier rejection"""

        self.reprojection_stiffness = np.diag([1., 1., 1.])
        """Reprojection error stiffness matrix"""
        self.mode = 'map'
        """Create new keyframes or localize against existing ones? ['map'|'track']"""

        self.loss = L2Loss()
        """Loss function"""

    def set_mode(self, mode):
        """Set the localization mode to ['map'|'track']"""
        self.mode = mode
        if self.mode == 'track':
            self.active_keyframe_idx = 0
            self.T_c_w = []

    def track(self, trackframe):
        """ Track an image.

            Args:
                trackframe  : frame to track
        """
        if len(self.keyframes) == 0:
            # First frame, so don't track anything yet
            self.keyframes.append(trackframe)
            self.active_keyframe_idx = 0
            active_keyframe = self.keyframes[0]
        else:
            # Default behaviour for second frame and beyond
            active_keyframe = self.keyframes[self.active_keyframe_idx]

            # Estimate pose change from keyframe to tracking frame
            T_track_ref = self._compute_frame_to_frame_motion(
                active_keyframe, trackframe)
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
                    self.keyframes.append(trackframe)

                    print('Dropped new keyframe. '
                          'Trans dist was {:.3f}. Rot dist was {:.3f}.'.format(
                              trans_dist, rot_dist))

                self.active_keyframe_idx += 1
                print('Active keyframe idx: {}'.format(
                    self.active_keyframe_idx))


class SparseStereoPipeline(SparseVOPipeline):
    """Sparse stereo VO pipeine"""

    def __init__(self, camera, first_pose=SE3.identity()):
        super().__init__(camera, first_pose)
        self.matcher_mode = 2  # stereo quad matching
        self.matcher.setIntrinsics(camera.fu, camera.cu, camera.cv, camera.b)

    def track(self, im_left, im_right):
        if len(self.keyframes) == 0:
            # First frame, so create first keyframe with given initial pose
            trackframe = SparseStereoKeyframe(im_left, im_right, self.T_c_w[0])
        else:
            # Default behaviour for second frame and beyond
            trackframe = SparseStereoKeyframe(im_left, im_right)

        super().track(trackframe)

    def _compute_frame_to_frame_motion(self, ref_frame, track_frame):
        # Get feature matches
        self.matcher.pushBack(ref_frame.im_left, ref_frame.im_right)
        self.matcher.pushBack(track_frame.im_left, track_frame.im_right)
        self.matcher.matchFeatures(self.matcher_mode)
        matches = self.matcher.getMatches()
        # print('libviso2 matched {} features.'.format(matches.size()))

        # Stereo observations (uvd)
        self.obs_0 = [np.array([m.u1p, m.v1p, m.u1p - m.u2p]) for m in matches]
        self.obs_1 = [np.array([m.u1c, m.v1c, m.u1c - m.u2c]) for m in matches]

        # Prune all observations with disparity <= 0
        keep_mask = (self.obs_0[:, 2] > 0) & (self.obs_1[:, 2] > 0)
        self.obs_0 = self.obs_0[keep_mask, :]
        self.obs_1 = self.obs_1[keep_mask, :]
        # print('Matches after pruning: {} '.format(len(self.obs_0)))

        # RANSAC
        self.ransac.set_obs(self.obs_0, self.obs_1)
        T_1_0_guess, obs_0_inliers, obs_1_inliers, _ = self.ransac.perform_ransac()

        # Optimize with inliers
        residual = ReprojectionMotionOnlyBatchResidual(
            self.camera, obs_0_inliers, obs_1_inliers, self.reprojection_stiffness)

        problem = Problem(self.motion_options)
        problem.add_residual_block(residual, ['T_1_0'], loss=self.loss)

        params = {'T_1_0': T_1_0_guess}
        problem.initialize_params(params)
        params = problem.solve()

        return params['T_1_0']


class SparseRGBDPipeline(SparseVOPipeline):
    """Sparse RGBD VO pipeline"""

    def __init__(self, camera, first_pose=SE3.identity()):
        super().__init__(camera, first_pose)
        self.matcher_mode = 0  # mono-to-mono

    def track(self, image, depth):
        if len(self.keyframes) == 0:
            # First frame, so create first keyframe with given initial pose
            trackframe = SparseRGBDKeyframe(image, depth, self.T_c_w[0])
        else:
            # Default behaviour for second frame and beyond
            trackframe = SparseRGBDKeyframe(image, depth)

        super().track(trackframe)

    def _compute_frame_to_frame_motion(self, ref_frame, track_frame):
        # Get feature matches
        self.matcher.pushBack(ref_frame.image)
        self.matcher.pushBack(track_frame.image)
        self.matcher.matchFeatures(self.matcher_mode)
        matches = self.matcher.getMatches()
        # print('libviso2 matched {} features.'.format(matches.size()))

        # RGB-D observations (uvz)
        self.obs_0 = np.array(
            [[m.u1p, m.v1p, ref_frame.depth[int(m.v1p), int(m.u1p)]] for m in matches])
        self.obs_1 = np.array(
            [[m.u1c, m.v1c, track_frame.depth[int(m.v1c), int(m.u1c)]] for m in matches])

        # Prune all observations with depth <= 0
        keep_mask = (self.obs_0[:, 2] > 0) & (self.obs_1[:, 2] > 0)
        self.obs_0 = self.obs_0[keep_mask, :]
        self.obs_1 = self.obs_1[keep_mask, :]
        # print('Matches after pruning: {} '.format(self.obs_0.shape[0]))

        # RANSAC
        self.ransac.set_obs(self.obs_0, self.obs_1)
        T_1_0_guess, obs_0_inliers, obs_1_inliers, _ = self.ransac.perform_ransac()

        # Optimize with inliers
        residual = ReprojectionMotionOnlyBatchResidual(
            self.camera, obs_0_inliers, obs_1_inliers, self.reprojection_stiffness)

        problem = Problem(self.motion_options)
        problem.add_residual_block(residual, ['T_1_0'], loss=self.loss)

        params = {'T_1_0': T_1_0_guess}
        problem.initialize_params(params)
        params = problem.solve()

        return params['T_1_0']
