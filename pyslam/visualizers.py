import numpy as np

import matplotlib
matplotlib.use('Agg') #Removes the XWindows backend (useful for producing plots via tmux without -X)
import matplotlib.pyplot as plt
from matplotlib import rc

from pyslam.metrics import TrajectoryMetrics

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class TrajectoryVisualizer:
    """Visualization utility to create nice plots from TrajectoryMetrics."""

    def __init__(self, tm_dict):
        self.tm_dict = tm_dict
        """Dictionary of TrajectoryMetrics objects to plot."""

    def plot_topdown(self, segment_range=None, outfile=None, **kwargs):
        """ Plot a top - down view of the trajectory.

            Args:
                segment_range   : which segment of the trajectory to plot
                outfile         : full path and filename where the plot should be saved
                **kwargs        : additional keyword arguments passed to plt.subplots()
        """
        fig, ax = plt.subplots(**kwargs)

        plotted_gt = False

        for label, tm in self.tm_dict.items():
            pos_gt = np.array([T.trans for T in tm.Twv_gt])
            pos_est = np.array([T.trans for T in tm.Twv_est])

            if segment_range is not None:
                pos_gt = pos_gt[segment_range, :]
                pos_est = pos_est[segment_range, :]

            if not plotted_gt:
                ax.plot(pos_gt[:, 0], pos_gt[:, 1], '-k',
                        linewidth=2, label='Ground Truth')
                plotted_gt = True

            ax.plot(pos_est[:, 0], pos_est[:, 1], label=label)

        ax.axis('equal')
        ax.minorticks_on()
        ax.grid(which='both', linestyle=':', linewidth=0.2)
        ax.set_title('Trajectory')
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.legend()

        if outfile is not None:
            print('Saving to {}'.format(outfile))
            fig.savefig(outfile)

        return fig, ax

    def plot_segment_errors(self, segs, outfile=None, **kwargs):
        """ Plot average errors for specified segment lengths, 
            similar to the KITTI odometry leaderboard.

            Args:
                segs    : list of segment lengths to plot
                outfile : full path and filename where the plot should be saved
                **kwargs: additional keyword arguments passed to plt.subplots()
        """
        # Use a sane default figsize if the user doesn't specify one
        figsize = kwargs.get('figsize', (12, 4))
        fig, ax = plt.subplots(1, 2, **dict(kwargs, figsize=figsize))

        for label, tm in self.tm_dict.items():
            segerr, avg_segerr = tm.segment_errors(segs)

            ax[0].plot(avg_segerr[:, 0], avg_segerr[:, 1]
                       * 100., '-s', label=label)
            ax[1].plot(avg_segerr[:, 0], avg_segerr[:, 2]
                       * 180. / np.pi, '-s', label=label)

        ax[0].minorticks_on()
        ax[0].grid(which='both', linestyle=':', linewidth=0.2)
        ax[0].set_title('Translational error')
        ax[0].set_xlabel('Segment length (m)')
        ax[0].set_ylabel('Average error (\%)')

        ax[1].minorticks_on()
        ax[1].grid(which='both', linestyle=':', linewidth=0.2)
        ax[1].set_title('Rotational error')
        ax[1].set_xlabel('Segment length (m)')
        ax[1].set_ylabel('Average error (deg/m)')
        ax[1].legend()

        if outfile is not None:
            print('Saving to {}'.format(outfile))
            fig.savefig(outfile)

        return fig, ax

    def _trans_rot_err_subplot(self, err_type, segment_range, outfile, **kwargs):
        """ Convenience function to plot translational and rotational errors in subplots.

            Args:
                err_type        : 'rmse', or 'crmse'
                segment_range   : which segment of the trajectory to plot
                outfile         : full path and filename where the plot should be saved
                **kwargs        : additional keyword arguments passed to plt.subplots()
        """
        # Use a sane default figsize if the user doesn't specify one
        figsize = kwargs.get('figsize', (12, 4))
        fig, ax = plt.subplots(1, 2, **dict(kwargs, figsize=figsize))

        for label, tm in self.tm_dict.items():
            if segment_range is None:
                segment_range = range(len(tm.Twv_gt))

            if err_type == 'mean':
                trans_err, rot_err = tm.mean_err(segment_range)
                err_name = 'Mean Err. Norm.'
            elif err_type == 'cum':
                trans_err, rot_err = tm.cum_err(segment_range)
                err_name = 'Cumulative Err. Norm.'
            else:
                raise ValueError(
                    'Got invalid err_type \'{}\''.format(err_type))

            ax[0].plot(trans_err, '-', label=label)
            ax[1].plot(rot_err * 180. / np.pi, '-', label=label)

        ax[0].minorticks_on()
        ax[0].grid(which='both', linestyle=':', linewidth=0.2)
        # ax[0].set_xlim((segment_range.start, segment_range.stop - 1))
        ax[0].set_title('Translational {}'.format(err_name))
        ax[0].set_xlabel('Timestep')
        ax[0].set_ylabel('{} (m)'.format(err_name))

        ax[1].minorticks_on()
        ax[1].grid(which='both', linestyle=':', linewidth=0.2)
        # ax[1].set_xlim((segment_range.start, segment_range.stop - 1))
        ax[1].set_title('Rotational {}'.format(err_name))
        ax[1].set_xlabel('Timestep')
        ax[1].set_ylabel('{} (deg)'.format(err_name))
        ax[1].legend()

        if outfile is not None:
            print('Saving to {}'.format(outfile))
            fig.savefig(outfile)

        return fig, ax

    def plot_pose_errors(self, segment_range=None, outfile=None, **kwargs):
        """ Plot translational and rotational errors over the trajectory.

            Args:
                segment_range   : which segment of the trajectory to plot
                outfile         : full path and filename where the plot should be saved
                **kwargs        : additional keyword arguments passed to plt.subplots()
        """
        return self._trans_rot_err_subplot('all', segment_range, outfile, **kwargs)

    def plot_mean_err(self, segment_range=None, outfile=None, **kwargs):
        """ Plot translational and rotational error norms over the trajectory.

            Args:
                segment_range   : which segment of the trajectory to plot
                outfile         : full path and filename where the plot should be saved
                **kwargs        : additional keyword arguments passed to plt.subplots()
        """
        return self._trans_rot_err_subplot('mean', segment_range, outfile, **kwargs)

    def plot_cum_err(self, segment_range=None, outfile=None, **kwargs):
        """ Plot cmulative translational and rotational error norms over the trajectory.

            Args:
                segment_range   : which segment of the trajectory to plot
                outfile         : full path and filename where the plot should be saved
                **kwargs        : additional keyword arguments passed to plt.subplots()
        """
        return self._trans_rot_err_subplot('cum', segment_range, outfile, **kwargs)
