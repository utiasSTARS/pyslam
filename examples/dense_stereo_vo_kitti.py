
import os
import time
from pyslam.visualizers import TrajectoryVisualizer
from pyslam.metrics import TrajectoryMetrics
from pyslam.sensors import StereoCamera
from pyslam.pipelines import DenseStereoPipeline
from liegroups import SE3
import pykitti
import numpy as np
import matplotlib
matplotlib.use('Agg')


def run_vo_kitti(basedir, date, drive, frames, outfile=None):
    # Load KITTI data
    dataset = pykitti.raw(basedir, date, drive, frames=frames, imformat='cv2')

    first_oxts = dataset.oxts[0]
    T_cam0_imu = SE3.from_matrix(dataset.calib.T_cam0_imu)
    T_cam0_imu.normalize()
    T_0_w = T_cam0_imu.dot(
        SE3.from_matrix(first_oxts.T_w_imu).inv())
    T_0_w.normalize()

    # Create the camera
    test_im = np.array(next(dataset.cam0))
    fu = dataset.calib.K_cam0[0, 0]
    fv = dataset.calib.K_cam0[1, 1]
    cu = dataset.calib.K_cam0[0, 2]
    cv = dataset.calib.K_cam0[1, 2]
    b = dataset.calib.b_gray
    h, w = test_im.shape
    camera = StereoCamera(cu, cv, fu, fv, b, w, h)

    # Ground truth
    T_w_c_gt = [SE3.from_matrix(o.T_w_imu).dot(T_cam0_imu.inv())
                for o in dataset.oxts]

    # Pipeline
    vo = DenseStereoPipeline(camera, first_pose=T_0_w)
    # Skip the highest resolution level
    vo.pyrlevel_sequence.pop()
    vo.pyr_cameras.pop()

    start = time.perf_counter()
    for c_idx, impair in enumerate(dataset.gray):
        vo.track(np.array(impair[0]), np.array(impair[1]))
        # vo.track(impair[0], impair[1], guess=T_w_c_gt[c_idx].inv())
        end = time.perf_counter()
        print('Image {}/{} | {:.3f} s'.format(c_idx, len(dataset), end - start))
        start = end

    # Compute errors
    T_w_c_est = [T.inv() for T in vo.T_c_w]
    tm = TrajectoryMetrics(T_w_c_gt, T_w_c_est)

    # Save to file
    if outfile is not None:
        print('Saving to {}'.format(outfile))
        tm.savemat(outfile)

    # Clean up
    del vo

    return tm


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

    basedir = '/path/to/KITTI/raw/'
    outdir = '/path_to/output/'
    os.makedirs(outdir, exist_ok=True)

    seqs = {
        '00': {'date': '2011_10_03',
               'drive': '0027',
               'frames': range(0, 4541)},
        '01': {'date': '2011_10_03',
               'drive': '0042',
               'frames': range(0, 1101)},
        '02': {'date': '2011_10_03',
               'drive': '0034',
               'frames': range(0, 4661)},
        '04': {'date': '2011_09_30',
               'drive': '0016',
               'frames': range(0, 271)},
        '05': {'date': '2011_09_30',
               'drive': '0018',
               'frames': range(0, 2761)},
        '06': {'date': '2011_09_30',
               'drive': '0020',
               'frames': range(0, 1101)},
        '07': {'date': '2011_09_30',
               'drive': '0027',
               'frames': range(0, 1101)},
        '08': {'date': '2011_09_30',
               'drive': '0028',
               'frames': range(1100, 5171)},
        '09': {'date': '2011_09_30',
               'drive': '0033',
               'frames': range(0, 1591)},
        '10': {'date': '2011_09_30',
               'drive': '0034',
               'frames': range(0, 1201)}
    }

    for key, val in seqs.items():
        date = val['date']
        drive = val['drive']
        frames = val['frames']

        print('Odometry sequence {} | {} {}'.format(key, date, drive))
        outfile = os.path.join(outdir, key + '.mat')
        tm = run_vo_kitti(basedir, date, drive, frames, outfile)

        # Compute errors
        trans_mean_err, rot_mean_err = tm.mean_err()
        print('trans mean err: {} meters'.format(trans_mean_err))
        print('rot mean err: {} deg'.format(rot_mean_err * 180. / np.pi))

        # Make plots
        visualizer = TrajectoryVisualizer({'VO': tm})

        outfile = os.path.join(outdir, key + '_err.pdf')
        segs = list(range(100, 801, 100))
        visualizer.plot_segment_errors(segs, outfile=outfile)

        outfile = os.path.join(outdir, key + '_traj.pdf')
        visualizer.plot_topdown(which_plane='xy', outfile=outfile)

        # Clean up
        del visualizer, tm


# Do the thing
main()
