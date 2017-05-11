import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import pykitti

from liegroups import SE3
from pyslam.pipelines import DenseStereoPipeline
from pyslam.sensors import StereoCamera
from pyslam.metrics import TrajectoryMetrics

import time
import os
import pickle


def run_vo_kitti(basedir, outdir, date, drive, frames):
    # Load KITTI data
    dataset = pykitti.raw(basedir, date, drive, frames=frames, imformat='cv2')

    first_oxts = next(dataset.oxts)
    T_cam0_imu = SE3.from_matrix(dataset.calib.T_cam0_imu)
    T_cam0_imu.normalize()
    T_0_w = T_cam0_imu * \
        SE3.from_matrix(first_oxts.T_w_imu).inv()
    T_0_w.normalize()

    # Create the camera
    test_im = next(dataset.cam0)
    fu = dataset.calib.K_cam0[0, 0]
    fv = dataset.calib.K_cam0[1, 1]
    cu = dataset.calib.K_cam0[0, 2]
    cv = dataset.calib.K_cam0[1, 2]
    b = dataset.calib.b_gray
    h, w = test_im.shape
    camera = StereoCamera(cu, cv, fu, fv, b, w, h)

    # Ground truth
    T_w_c_gt = [SE3.from_matrix(o.T_w_imu) * T_cam0_imu.inv()
                for o in dataset.oxts]

    # Pipeline
    vo = DenseStereoPipeline(camera, first_pose=T_0_w)
    start = time.perf_counter()
    for c_idx, impair in enumerate(dataset.gray):
        vo.track(impair[0], impair[1])
        # vo.track(impair[0], impair[1], guess=T_w_c_gt[c_idx].inv())
        end = time.perf_counter()
        print('Image {}/{} | {:.3f} s'.format(c_idx, len(dataset), end - start))
        start = end

    # Compute errors
    T_w_c_est = [T.inv() for T in vo.T_c_w]

    tm = TrajectoryMetrics(T_w_c_gt, T_w_c_est)

    trans_armse, rot_armse = tm.armse()
    print('trans armse: {} meters'.format(trans_armse))
    print('rot armse: {} deg'.format(rot_armse * 180. / np.pi))

    # Output to file
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, date + '_drive_' + drive + '.pickle')

    with open(outfile, 'wb') as f:
        pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)

    # Make plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    segs = np.linspace(100, 800, 8)
    pos_gt = np.array([T.trans for T in tm.Twv_gt])
    pos_est = np.array([T.trans for T in tm.Twv_est])

    f_err, ax_err = plt.subplots(1, 2, figsize=(12, 4))
    f_traj, ax_traj = plt.subplots()

    plt.plot(pos_gt[:, 0], pos_gt[:, 1], '-k',
             linewidth=2, label='Ground Truth')

    ax_traj.plot(pos_est[:, 0], pos_est[:, 1], label='VO')

    segerr, avg_segerr = tm.segment_errors(segs)

    ax_err[0].plot(avg_segerr[:, 0], avg_segerr[:, 1]
                   * 100., '-s')
    ax_err[1].plot(avg_segerr[:, 0], avg_segerr[:, 2]
                   * 180. / np.pi, '-s')

    ax_traj.axis('equal')
    ax_traj.minorticks_on()
    ax_traj.grid(which='both', linestyle=':', linewidth=0.2)
    ax_traj.set_title('Trajectory')
    ax_traj.set_xlabel('Easting (m)')
    ax_traj.set_ylabel('Northing (m)')
    ax_traj.legend()

    ax_err[0].minorticks_on()
    ax_err[0].grid(which='both', linestyle=':', linewidth=0.2)
    ax_err[0].set_title('Translational error')
    ax_err[0].set_xlabel('Sequence length (m)')
    ax_err[0].set_ylabel('Average error (\%)')

    ax_err[1].minorticks_on()
    ax_err[1].grid(which='both', linestyle=':', linewidth=0.2)
    ax_err[1].set_title('Rotational error')
    ax_err[1].set_xlabel('Sequence length (m)')
    ax_err[1].set_ylabel('Average error (deg/m)')

    f_err.savefig(os.path.join(outdir, date + '_drive_' + drive + '_err.pdf'))
    f_traj.savefig(os.path.join(
        outdir, date + '_drive_' + drive + '_traj.pdf'))


def main():
    # Odometry sequences
    # Nr.     Sequence name     Start   End
    # ---------------------------------------
    # 00: 2011_10_03_drive_0027 000000 004540
    # 01: 2011_10_03_drive_0042 000000 001100
    # 02: 2011_10_03_drive_0034 000000 004660
    # 03: 2011_09_26_drive_0067 000000 000800
    # 04: 2011_09_30_drive_0016 000000 000270
    # 05: 2011_09_30_drive_0018 000000 002760
    # 06: 2011_09_30_drive_0020 000000 001100
    # 07: 2011_09_30_drive_0027 000000 001100
    # 08: 2011_09_30_drive_0028 001100 005170
    # 09: 2011_09_30_drive_0033 000000 001590
    # 10: 2011_09_30_drive_0034 000000 001200

    basedir = '/Users/leeclement/Desktop/KITTI/raw/'
    outdir = '/Users/leeclement/Desktop/pyslam/KITTI/'

    seqs = {'00': {'date': '2011_10_03',
                   'drive': '0027',
                   'frames': range(0, 4540)},
            '01': {'date': '2011_10_03',
                   'drive': '0042',
                   'frames': range(0, 1100)},
            '02': {'date': '2011_10_03',
                   'drive': '0034',
                   'frames': range(0, 4660)},
            '04': {'date': '2011_09_30',
                   'drive': '0016',
                   'frames': range(0, 270)},
            '05': {'date': '2011_09_30',
                   'drive': '0018',
                   'frames': range(0, 2760)},
            '06': {'date': '2011_09_30',
                   'drive': '0020',
                   'frames': range(0, 1100)},
            '07': {'date': '2011_09_30',
                   'drive': '0027',
                   'frames': range(0, 1100)},
            '08': {'date': '2011_09_30',
                   'drive': '0028',
                   'frames': range(1100, 5170)},
            '09': {'date': '2011_09_30',
                   'drive': '0033',
                   'frames': range(0, 1590)},
            '10': {'date': '2011_09_30',
                   'drive': '0034',
                   'frames': range(0, 1200)}}

    for key, val in seqs.items():
        date = val['date']
        drive = val['drive']
        frames = val['frames']

        print('Odometry sequence {} | {} {}'.format(key, date, drive))
        run_vo_kitti(basedir, outdir, date, drive, frames)


main()