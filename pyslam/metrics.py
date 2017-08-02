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

    def savemat(self, filename):
        """Save trajectory data to a .mat file.

           Saved poses will use the same convention as the input poses.
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
        scipy.io.savemat(filename, mdict, do_compression=True)

    @classmethod
    def loadmat(cls, filename):
        """Load trajectory data from a .mat file."""
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
        poses_gt = [pose_type.from_matrix(poses_gt[:, :, i])
                    for i in range(num_poses)]
        poses_est = [pose_type.from_matrix(poses_est[:, :, i])
                     for i in range(num_poses)]

        return cls(poses_gt, poses_est, convention=mdict['convention'])

    def endpoint_error(self, segment_range=None):
        """Returns translational and rotational error at the endpoint of a segment"""
        if segment_range is None:
            segment_range = range(len(self.Twv_gt))

        pose_delta_gt = self.Twv_gt[segment_range[0]].inv().dot(
            self.Twv_gt[segment_range[-1]])
        pose_delta_est = self.Twv_est[segment_range[0]].inv().dot(
            self.Twv_est[segment_range[-1]])

        pose_error = self.pose_type.log(
            pose_delta_est.inv().dot(pose_delta_gt))
        trans_err = np.linalg.norm(pose_error[0:3])
        rot_err = np.linalg.norm(pose_error[3:6])

        return trans_err, rot_err

    def segment_errors(self, segment_lengths):
        """Compute endpoint errors and average endpoint errors
            all possible segments of specified lengths in meters.

            Output format (Nx3):
            length | proportional trans err (unitless) | proportional rot err (rad/meter)
        """
        # Compute all endpoint errors for each segment length
        errs = []
        for length in segment_lengths:
            for start in range(self.num_poses):
                # Find the index of the pose s.t. distance relative to segment
                # start is >= length
                stop = np.searchsorted(
                    self.cum_dists - self.cum_dists[start],
                    length, side='right')

                # stop == self.num_poses means no solution
                if stop < self.num_poses:
                    trans_err, rot_err = self.endpoint_error(
                        range(start, stop + 1))
                    errs.append([length, trans_err / length, rot_err / length])

        errs = np.array(errs)

        # Compute average endpoint error for each segment length
        avg_errs = []
        for length in segment_lengths:
            avg_errs.append(np.mean(errs[errs[:, 0] == length], axis=0))

        avg_errs = np.array(avg_errs)

        return errs, avg_errs

    def pose_errors(self, segment_range=None):
        """Returns translational (m) and rotational (rad) errors
            in all degrees of freedom
        """
        if segment_range is None:
            segment_range = range(len(self.Twv_gt))

        errs = []
        for p_idx in segment_range:
            pose_delta_gt = self.Twv_gt[segment_range[0]].inv().dot(
                self.Twv_gt[p_idx])
            pose_delta_est = self.Twv_est[segment_range[0]].inv().dot(
                self.Twv_est[p_idx])
            errs.append(self.pose_type.log(
                pose_delta_est.inv().dot(pose_delta_gt)))

        errs = np.array(errs)
        trans_err = errs[:, 0:3]
        rot_err = errs[:, 3:6]

        return trans_err, rot_err

    def rmse(self, segment_range=None):
        """Root mean squared error (RMSE) of the trajectory."""
        trans_errs, rot_errs = self.pose_errors(segment_range)

        trans_rmse = np.sqrt(np.mean(trans_errs**2, axis=1))
        rot_rmse = np.sqrt(np.mean(rot_errs**2, axis=1))

        return trans_rmse, rot_rmse

    def armse(self, segment_range=None):
        """Average root mean squared error (ARMSE) of the trajectory."""
        trans_rmse, rot_rmse = self.rmse(segment_range)
        return np.mean(trans_rmse), np.mean(rot_rmse)

    def crmse(self, segment_range=None):
        """Cumulative root mean squared error (CRMSE) of the trajectory."""
        trans_rmse, rot_rmse = self.rmse(segment_range)
        return np.cumsum(trans_rmse), np.cumsum(rot_rmse)
