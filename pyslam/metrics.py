import numpy as np
import scipy.io

from liegroups import SE2, SE3


class TrajectoryMetrics:
    """Class for computing metrics on SE2/SE3 trajectories.

        convention='Twv' -- poses are vehicle-to-world transforms
        convention='Tvw' -- poses are world-to-vehicle transforms
                            (will be converted to Twv internally)
    """

    def __init__(self, poses_gt, poses_est, convention='Twv'):
        if convention == 'Twv':
            Twv_gt = poses_gt
            Twv_est = poses_est
        elif convention == 'Tvw':
            Twv_gt = [T.inv() for T in poses_gt]
            Twv_est = [T.inv() for T in poses_est]
        else:
            raise ValueError('convention must be \'Tvw\' or \'Twv\'')

        if len(Twv_gt) != len(Twv_est):
            valid_length = min((len(Twv_gt), len(Twv_est)))

            print('WARNING: poses_gt has length {} but poses_est has length {}. Truncating to {}.'.format(
                len(Twv_gt), len(Twv_est), valid_length))
            Twv_gt = Twv_gt[:valid_length]
            Twv_est = Twv_est[:valid_length]

        self.convention = convention
        """Were the input poses Twv or Tvw?"""
        self.Twv_gt = Twv_gt
        """List of ground truth vehicle-to-world poses."""
        self.Twv_est = Twv_est
        """List of estimated vehicle-to-world poses."""
        self.pose_type = type(Twv_gt[0])
        """SE2 or SE3?"""

        self.num_poses = len(self.Twv_gt)
        """Number of poses"""

        self.rel_dists, self.cum_dists = self._compute_distances()
        """Relative and cumulative distances traveled at each pose"""

    def _compute_distances(self):
        """Returns relative and cumulative distances traveled at each pose"""
        pos_gt = np.empty([len(self.Twv_gt), 3])
        for p_idx, Twv in enumerate(self.Twv_gt):
            pos_gt[p_idx, :] = Twv.trans

        rel_dist_gt = np.linalg.norm(np.diff(pos_gt[:, 0:3], axis=0), axis=1)
        rel_dist_gt = np.append([0.], rel_dist_gt)
        cum_dist_gt = np.cumsum(rel_dist_gt)
        return rel_dist_gt, cum_dist_gt

    def _convert_meters(self, meters, unit):
        """Convert meters to unit ['m', 'dm', 'cm', 'mm']"""
        scale = {
            'm': 1.,
            'dm': 10.,
            'cm': 100.,
            'mm': 1000.
        }[unit]

        return scale * meters

    def _convert_radians(self, radians, unit):
        """Convert radians to unit ['rad', 'deg']"""
        scale = {
            'rad': 1,
            'deg': 180. / np.pi
        }[unit]

        return scale * radians

    def savemat(self, filename, extras=None):
        """Save trajectory data to a .mat file.
           Saved poses will use the same convention as the input poses.

            Args:
                filename : path of file to save
                extras   : optional dictionary of extra arrays to save
        """
        # Convert poses to matrices using original convention
        if self.convention == 'Twv':
            poses_gt = [T.as_matrix() for T in self.Twv_gt]
            poses_est = [T.as_matrix() for T in self.Twv_est]
        elif self.convention == 'Tvw':
            poses_gt = [T.inv().as_matrix() for T in self.Twv_gt]
            poses_est = [T.inv().as_matrix() for T in self.Twv_est]

        # Permute the dimensions to native MATLAB ordering (NxMxM --> MxMxN)
        poses_gt = np.transpose(np.array(poses_gt), [1, 2, 0])
        poses_est = np.transpose(np.array(poses_est), [1, 2, 0])

        # Save to file
        mdict = {'poses_gt': poses_gt,
                 'poses_est': poses_est,
                 'convention': self.convention,
                 'pose_type': self.pose_type.__name__,
                 'num_poses': self.num_poses,
                 'rel_dists': self.rel_dists,
                 'cum_dists': self.cum_dists}
        if extras is not None:
            mdict.update(extras)

        scipy.io.savemat(filename, mdict, do_compression=True)

    @classmethod
    def loadmat(cls, filename):
        """Load trajectory data from a .mat file.

            Args:
                filename : path of file to load
        """
        # Load from file
        mdict = scipy.io.loadmat(
            filename, verify_compressed_data_integrity=True)

        # Unpack data
        if mdict['pose_type'] == 'SE2':
            pose_type = SE2
        elif mdict['pose_type'] == 'SE3':
            pose_type = SE3
        else:
            raise ValueError(
                'Got invalid pose type: {}'.format(mdict['pose_type']))

        num_poses = int(mdict['num_poses'])
        poses_gt = mdict['poses_gt']
        poses_est = mdict['poses_est']

        # Convert to liegroups objects and create TrajectoryMetrics object
        poses_gt = [pose_type.from_matrix(poses_gt[:, :, i], normalize=True)
                    for i in range(num_poses)]
        poses_est = [pose_type.from_matrix(poses_est[:, :, i], normalize=True)
                     for i in range(num_poses)]

        tm = cls(poses_gt, poses_est, convention=mdict['convention'])
        tm.mdict = mdict

        return tm

    def endpoint_error(self, segment_range=None, trans_unit='m', rot_unit='rad'):
        """Returns translational and rotational error at the endpoint of a segment"""
        if segment_range is None:
            segment_range = range(len(self.Twv_gt))

        pose_delta_gt = self.Twv_gt[segment_range[0]].inv().dot(
            self.Twv_gt[segment_range[-1]])
        pose_delta_est = self.Twv_est[segment_range[0]].inv().dot(
            self.Twv_est[segment_range[-1]])

        pose_err = pose_delta_est.inv().dot(pose_delta_gt)
        trans_err = np.linalg.norm(pose_err.trans)
        rot_err = np.linalg.norm(pose_err.rot.log())

        return self._convert_meters(trans_err, trans_unit), self._convert_radians(rot_err, rot_unit)

    def segment_errors(self, segment_lengths, trans_unit='m', rot_unit='rad'):
        """Compute endpoint errors and average endpoint errors
            all possible segments of specified lengths in meters.

            Output format (Nx3):
            length | proportional trans err (unitless) | proportional rot err (rad/meter)
        """
        # Compute all endpoint errors for each segment length
        errs = []
        for length in segment_lengths:
            length = self._convert_meters(length, trans_unit)

            for start in range(self.num_poses):
                # Find the index of the pose s.t. distance relative to segment
                # start is >= length
                stop = np.searchsorted(
                    self.cum_dists - self.cum_dists[start],
                    length, side='right')

                # stop == self.num_poses means no solution
                if stop < self.num_poses:
                    trans_err, rot_err = self.endpoint_error(
                        range(start, stop + 1), trans_unit, rot_unit)

                    errs.append([length, trans_err / length, rot_err / length])

        errs = np.array(errs)

        # Compute average endpoint error for each segment length
        avg_errs = []
        for length in segment_lengths:
            length = self._convert_meters(length, trans_unit)
            avg_errs.append(np.mean(errs[errs[:, 0] == length], axis=0))

        avg_errs = np.array(avg_errs)

        return errs, avg_errs

    def traj_errors(self, segment_range=None, trans_unit='m', rot_unit='rad'):
        """Returns translational (m) and rotational (rad) errors
            in all degrees of freedom
        """
        if segment_range is None:
            segment_range = range(len(self.Twv_gt))

        trans_err = []
        rot_err = []

        for p_idx in segment_range:
            pose_delta_gt = self.Twv_gt[segment_range[0]].inv().dot(
                self.Twv_gt[p_idx])
            pose_delta_est = self.Twv_gt[segment_range[0]].inv().dot(
                self.Twv_est[p_idx])

            pose_err = pose_delta_est.inv().dot(pose_delta_gt)
            trans_err.append(pose_err.trans)
            rot_err.append(pose_err.rot.log())

        trans_err = np.array(trans_err)
        rot_err = np.array(rot_err)

        return self._convert_meters(trans_err, trans_unit), self._convert_radians(rot_err, rot_unit)

    def rel_errors(self, segment_range=None, trans_unit='m', rot_unit='rad', delta=1):
        """Returns translational (m) and rotational (rad) relative pose errors (RPEs)
            in all degrees of freedom - See equation (1) in "A Benchmark for the Evaluation of RGB-D SLAM Systems" by Sturm et al.
        """
        if segment_range is None:
            segment_range = range(len(self.Twv_gt))

        trans_err = []
        rot_err = []

        for p_idx in segment_range[:-delta]:
            rel_pose_delta_gt = self.Twv_gt[p_idx].inv().dot(
                self.Twv_gt[p_idx + delta])
            rel_pose_delta_est = self.Twv_est[p_idx].inv().dot(
                self.Twv_est[p_idx + delta])

            pose_err = rel_pose_delta_gt.inv().dot(rel_pose_delta_est)
            trans_err.append(pose_err.trans)
            rot_err.append(pose_err.rot.log())

        trans_err = np.array(trans_err)
        rot_err = np.array(rot_err)

        return self._convert_meters(trans_err, trans_unit), self._convert_radians(rot_err, rot_unit)

    def error_norms(self, segment_range=None, trans_unit='m', rot_unit='rad', error_type='traj', delta=1):
        """Error norms (magnitude of errors in rotation and translation) of the trajectory."""

        if error_type == 'traj':
            trans_errs, rot_errs = self.traj_errors(
                segment_range, trans_unit, rot_unit)
        elif error_type == 'rel':
            trans_errs, rot_errs = self.rel_errors(
                segment_range, trans_unit, rot_unit, delta)
        else:
            raise ValueError('error_type must be either `traj` or `rel`.')

        trans_norms = np.sqrt(np.sum(trans_errs**2, axis=1))
        rot_norms = np.sqrt(np.sum(rot_errs**2, axis=1))

        return trans_norms, rot_norms

    def mean_err(self, segment_range=None, trans_unit='m', rot_unit='rad', error_type='traj'):
        """Mean of the rotation and translation error magnitudes over the entire trajectory.

            Notes:
            error_type='traj' computes errors relative to ground truth for N T_wv poses (with respect to T_wv[0])
            error_type='rel' computes errors relative to ground truth over N-1 consecutive frame-to-frame transforms

        """
        trans_norms, rot_norms = self.error_norms(
            segment_range, trans_unit, rot_unit, error_type)
        return np.mean(trans_norms), np.mean(rot_norms)

    def cum_err(self, segment_range=None, trans_unit='m', rot_unit='rad', error_type='traj'):
        """Cumulative sum of the rotation and translation error magnitudes over the entire trajectory.

            Notes:
            error_type='traj' computes errors relative to ground truth for N T_wv poses (with respect to T_wv[0])
            error_type='rel' computes errors relative to ground truth over N-1 consecutive frame-to-frame transforms
        """
        trans_norms, rot_norms = self.error_norms(
            segment_range, trans_unit, rot_unit, error_type)
        return np.cumsum(trans_norms), np.cumsum(rot_norms)

    def rms_err(self, segment_range=None, trans_unit='m', rot_unit='rad', error_type='traj', delta=1):
        """RMS of the rotation and translation error magnitudes over the entire trajectory.

            Notes:
            error_type='traj' computes errors relative to ground truth for N T_wv poses (with respect to T_wv[0])
            error_type='rel' computes errors relative to ground truth over N-1 consecutive frame-to-frame transforms
        """
        trans_norms, rot_norms = self.error_norms(
            segment_range, trans_unit, rot_unit, error_type, delta)
        return np.sqrt(np.mean(trans_norms**2)), np.sqrt(np.mean(rot_norms**2))
