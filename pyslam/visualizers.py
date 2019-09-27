from pyslam.metrics import TrajectoryMetrics
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
# Removes the XWindows backend (useful for producing plots via tmux without -X)
matplotlib.use('Agg', warn=False)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class TrajectoryVisualizer:
    """Visualization utility to create nice plots from TrajectoryMetrics."""

    def __init__(self, tm_dict):
        self.tm_dict = tm_dict
        """Dictionary of TrajectoryMetrics objects to plot."""

        self.endpoint_markers_list = ['o', '^', 's', '*', 'p', 'h',
                                      'X', 'D', 'P',  'v', '<', '>',
                                      '8', 'H', 'd']

    def _parse_kwargs(self, kwargs):
        plot_params = {}
        plot_params['gt_linewidth'] = kwargs.get('gt_linewidth', 2.)
        plot_params['est_linewidth'] = kwargs.get('est_linewidth', 1.)
        plot_params['grid_linewidth'] = kwargs.get('grid_linewidth', 0.2)
        plot_params['use_endpoint_markers'] = kwargs.get(
            'use_endpoint_markers', False)
        plot_params['fontsize'] = kwargs.get('fontsize', 10)
        plot_params['legend_fontsize'] = kwargs.get(
            'legend_fontsize', plot_params['fontsize'])
        plot_params['err_xlabel'] = kwargs.get('err_xlabel', 'Timestep')
        plot_params['line_colours'] = kwargs.get('line_colours', [
                                                 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'])

        for key in plot_params.keys():
            try:
                del kwargs[key]
            except KeyError:
                pass

        return plot_params

    def plot_topdown(self, which_plane='xz', segment_range=None, outfile=None, **kwargs):
        """ Plot a top - down view of the trajectory.

            Args:
                which_plane     : which plane to plot ['xy' | 'xz' | 'yz']
                segment_range   : which segment of the trajectory to plot
                outfile         : full path and filename where the plot should be saved
                **kwargs        : additional keyword arguments passed to plt.subplots()
        """
        if which_plane == 'xy':
            x_dim, y_dim = 0, 1
        elif which_plane == 'xz':
            x_dim, y_dim = 0, 2
        elif which_plane == 'yz':
            x_dim, y_dim = 1, 2
        else:
            raise ValueError(
                "which_plane must be ['xy' | 'xz' | 'yz']. Got {}".format(which_plane))

        # Grab plot parameters, pass the rest to subplots
        plot_params = self._parse_kwargs(kwargs)

        # Use a sane default figsize if the user doesn't specify one
        figsize = kwargs.get('figsize', (4, 3))
        kwargs.update({'figsize': figsize})

        fig, ax = plt.subplots(**kwargs)

        if plot_params['use_endpoint_markers']:
            endpoint_markers_iter = iter(self.endpoint_markers_list)

        plotted_gt = False

        for p_i, (label, tm) in enumerate(self.tm_dict.items()):
            pos_gt = np.array([T.trans for T in tm.Twv_gt])
            pos_est = np.array([T.trans for T in tm.Twv_est])

            if segment_range is not None:
                pos_gt = pos_gt[segment_range, :]
                pos_est = pos_est[segment_range, :]

            if not plotted_gt:
                if plot_params['use_endpoint_markers']:
                    ax.plot(pos_gt[:, x_dim], pos_gt[:, y_dim], '-k',
                            linewidth=plot_params['gt_linewidth'],
                            marker=next(endpoint_markers_iter), markevery=[pos_gt.shape[0] - 1],
                            label='Ground Truth')
                else:
                    ax.plot(pos_gt[:, x_dim], pos_gt[:, y_dim], '-k',
                            linewidth=plot_params['gt_linewidth'], label='Ground Truth')

                plotted_gt = True

            if plot_params['use_endpoint_markers']:
                ax.plot(pos_est[:, x_dim], pos_est[:, y_dim],
                        linewidth=plot_params['est_linewidth'], color=plot_params['line_colours'][p_i],
                        marker=next(endpoint_markers_iter), markevery=[pos_est.shape[0] - 1],
                        label=label)
            else:
                ax.plot(pos_est[:, x_dim], pos_est[:, y_dim], color=plot_params['line_colours'][p_i],
                        linewidth=plot_params['est_linewidth'], label=label)

        ax.axis('equal')
        # ax.minorticks_on()
        ax.grid(which='both', linestyle=':',
                linewidth=plot_params['grid_linewidth'])
        ax.set_title('Trajectory', fontsize=plot_params['fontsize'])
        ax.set_xlabel('Easting (m)', fontsize=plot_params['fontsize'])
        ax.set_ylabel('Northing (m)', fontsize=plot_params['fontsize'])
        ax.legend(fontsize=plot_params['legend_fontsize'])

        if outfile is not None:
            print('Saving to {}'.format(outfile))
            fig.savefig(outfile, bbox_inches='tight', transparent=True)

        return fig, ax

    def plot_segment_errors(self, segs, outfile=None, **kwargs):
        """ Plot average errors for specified segment lengths, 
            similar to the KITTI odometry leaderboard.

            Args:
                segs    : list of segment lengths to plot
                outfile : full path and filename where the plot should be saved
                **kwargs: additional keyword arguments passed to plt.subplots()
        """
        # Grab plot parameters, pass the rest to subplots
        plot_params = self._parse_kwargs(kwargs)

        # Use a sane default figsize if the user doesn't specify one
        figsize = kwargs.get('figsize', (8, 3))
        kwargs.update({'figsize': figsize})

        fig, ax = plt.subplots(1, 2, **kwargs)
        handles = []
        legend = []

        for p_i, (label, tm) in enumerate(self.tm_dict.items()):
            segerr, avg_segerr = tm.segment_errors(segs)

            h = ax[0].plot(avg_segerr[:, 0], avg_segerr[:, 1]
                           * 100., '-s', color=plot_params['line_colours'][p_i], label=label)
            ax[1].plot(avg_segerr[:, 0], avg_segerr[:, 2]
                       * 180. / np.pi, '-s', color=plot_params['line_colours'][p_i], label=label)

            handles.append(h[0])
            legend.append(label)

        ax[0].minorticks_on()
        ax[0].grid(which='both', linestyle=':',
                   linewidth=plot_params['grid_linewidth'])
        ax[0].set_title('Translational error',
                        fontsize=plot_params['fontsize'])
        ax[0].set_xlabel('Segment length (m)',
                         fontsize=plot_params['fontsize'])
        ax[0].set_ylabel('Avg. err. (\%)',
                         fontsize=plot_params['fontsize'])

        ax[1].minorticks_on()
        ax[1].grid(which='both', linestyle=':',
                   linewidth=plot_params['grid_linewidth'])
        ax[1].set_title('Rotational error',
                        fontsize=plot_params['fontsize'])
        ax[1].set_xlabel('Segment length (m)',
                         fontsize=plot_params['fontsize'])
        ax[1].set_ylabel('Avg. err. (deg/m)',
                         fontsize=plot_params['fontsize'])

        fig.legend(handles, legend, loc='lower center', ncol=len(self.tm_dict),
                   fontsize=plot_params['legend_fontsize'])
        fig.subplots_adjust(wspace=0.3, bottom=0.35)

        if outfile is not None:
            print('Saving to {}'.format(outfile))
            fig.savefig(outfile, bbox_inches='tight', transparent=True)

        return fig, ax

    def _trans_rot_err_subplot(self, err_type, segment_range, outfile, **kwargs):
        """ Convenience function to plot translational and rotational errors in subplots.

            Args:
                err_type        : 'rmse', or 'crmse'
                segment_range   : which segment of the trajectory to plot
                outfile         : full path and filename where the plot should be saved
                **kwargs        : additional keyword arguments passed to plt.subplots()
        """
        # Grab plot parameters, pass the rest to subplots
        plot_params = self._parse_kwargs(kwargs)

        # Use a sane default figsize if the user doesn't specify one
        figsize = kwargs.get('figsize', (8, 3))
        kwargs.update({'figsize': figsize})
        fig, ax = plt.subplots(1, 2, **kwargs)
        handles = []
        legend = []

        for p_i, (label, tm) in enumerate(self.tm_dict.items()):
            if segment_range is None:
                this_segment_range = range(len(tm.Twv_gt))

            if err_type == 'norm':
                trans_err, rot_err = tm.error_norms(this_segment_range)
                err_name = 'Err. Norm.'
            elif err_type == 'cum':
                trans_err, rot_err = tm.cum_err(this_segment_range)
                err_name = 'Cumulative Err. Norm.'
            else:
                raise ValueError(
                    'Got invalid err_type \'{}\''.format(err_type))

            h = ax[0].plot(trans_err, '-',
                           color=plot_params['line_colours'][p_i], label=label)
            ax[1].plot(rot_err * 180. / np.pi, '-',
                       color=plot_params['line_colours'][p_i], label=label)

            handles.append(h[0])
            legend.append(label)

        ax[0].minorticks_on()
        ax[0].grid(which='both', linestyle=':',
                   linewidth=plot_params['grid_linewidth'])
        # ax[0].set_xlim((segment_range.start, segment_range.stop - 1))
        ax[0].set_title('Translational {}'.format(err_name),
                        fontsize=plot_params['fontsize'])
        ax[0].set_xlabel(plot_params['err_xlabel'],
                         fontsize=plot_params['fontsize'])
        ax[0].set_ylabel('{} (m)'.format(err_name),
                         fontsize=plot_params['fontsize'])

        ax[1].minorticks_on()
        ax[1].grid(which='both', linestyle=':',
                   linewidth=plot_params['grid_linewidth'])
        # ax[1].set_xlim((segment_range.start, segment_range.stop - 1))
        ax[1].set_title('Rotational {}'.format(err_name),
                        fontsize=plot_params['fontsize'])
        ax[1].set_xlabel(plot_params['err_xlabel'],
                         fontsize=plot_params['fontsize'])
        ax[1].set_ylabel('{} (deg)'.format(err_name),
                         fontsize=plot_params['fontsize'])

        fig.legend(handles, legend, loc='lower center', ncol=len(self.tm_dict),
                   fontsize=plot_params['legend_fontsize'])
        fig.subplots_adjust(wspace=0.3, bottom=0.35)

        if outfile is not None:
            print('Saving to {}'.format(outfile))
            fig.savefig(outfile, bbox_inches='tight', transparent=True)

        return fig, ax

    def plot_pose_errors(self, segment_range=None, outfile=None, **kwargs):
        """ Plot translational and rotational errors over the trajectory.

            Args:
                segment_range   : which segment of the trajectory to plot
                outfile         : full path and filename where the plot should be saved
                **kwargs        : additional keyword arguments passed to plt.subplots()
        """
        return self._trans_rot_err_subplot('all', segment_range, outfile, **kwargs)

    def plot_norm_err(self, segment_range=None, outfile=None, **kwargs):
        """ Plot translational and rotational error norms over the trajectory.

            Args:
                segment_range   : which segment of the trajectory to plot
                outfile         : full path and filename where the plot should be saved
                **kwargs        : additional keyword arguments passed to plt.subplots()
        """
        return self._trans_rot_err_subplot('norm', segment_range, outfile, **kwargs)

    def plot_cum_norm_err(self, segment_range=None, outfile=None, **kwargs):
        """ Plot cmulative translational and rotational error norms over the trajectory.

            Args:
                segment_range   : which segment of the trajectory to plot
                outfile         : full path and filename where the plot should be saved
                **kwargs        : additional keyword arguments passed to plt.subplots()
        """
        return self._trans_rot_err_subplot('cum', segment_range, outfile, **kwargs)
