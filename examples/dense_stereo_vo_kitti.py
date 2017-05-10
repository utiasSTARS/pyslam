import numpy as np
import matplotlib.pyplot as plt

import pykitti

from liegroups import SE3
from pyslam.pipelines import DenseStereoPipeline
from pyslam.sensors import StereoCamera
from pyslam.metrics import TrajectoryMetrics

import time
import os
import pickle

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

# Load KITTI data
basedir = '/Users/leeclement/Desktop/KITTI/raw/'
date = '2011_09_30'
drive = '0033'
frames = range(0, 1590)

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
for c_idx, impair in enumerate(dataset.gray):
    print('Image {}'.format(c_idx))
    vo.track(impair[0], impair[1])
    # vo.track(impair[0], impair[1], T_w_c_gt[c_idx].inv())

# Compute errors
T_w_c_est = [T.inv() for T in vo.T_c_w]

tm = TrajectoryMetrics(T_w_c_gt, T_w_c_est)

# Output to file
outdir = '/Users/leeclement/Desktop/pyslam/KITTI/'
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, date + '_drive_' + drive + '.pickle')

with open(outfile, 'wb') as f:
    pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)


# Make plots
pos_gt = np.array([T.trans for T in tm.Twv_gt])
pos_est = np.array([T.trans for T in tm.Twv_est])

plt.figure()
plt.plot(pos_gt[:, 0], pos_gt[:, 1], label='gt')
plt.plot(pos_est[:, 0], pos_est[:, 1], label='est')
plt.axis('equal')
plt.legend()

segerrs, avg_segerr = tm.segment_errors(np.linspace(100, 800, 8))
# segerrs, avg_segerr = tm.segment_errors([1, 5, 10])

f, ax = plt.subplots(1, 2)
ax[0].plot(avg_segerr[:, 0], avg_segerr[:, 1] * 100., '-s')
ax[0].set_title('Translational error')
ax[0].set_xlabel('Sequence length (m)')
ax[0].set_ylabel('Average error (%)')
ax[1].plot(avg_segerr[:, 0], avg_segerr[:, 2] * 180. / np.pi, '-s')
ax[1].set_title('Rotational error')
ax[1].set_xlabel('Sequence length (m)')
ax[1].set_ylabel('Average error (deg/m)')

# f, ax = plt.subplots(1, 2)
# ax[0].scatter(segerrs[:, 0], segerrs[:, 1] * 100.)
# ax[0].set_title('Translational error')
# ax[0].set_xlabel('Sequence length (m)')
# ax[0].set_ylabel('Error (%)')
# ax[1].scatter(segerrs[:, 0], segerrs[:, 2] * 180. / np.pi)
# ax[1].set_title('Rotational error')
# ax[1].set_xlabel('Sequence length (m)')
# ax[1].set_ylabel('Error (deg/m)')

# trans_errs, rot_errs = tm.pose_errors()
# plt.figure()
# plt.plot(tm.distances, trans_errs[:, 0], label='x')
# plt.plot(tm.distances, trans_errs[:, 1], label='y')
# plt.plot(tm.distances, trans_errs[:, 2], label='z')
# plt.plot(tm.distances, rot_errs[:, 0], label=r'$\theta_x$')
# plt.plot(tm.distances, rot_errs[:, 1], label=r'$\theta_y$')
# plt.plot(tm.distances, rot_errs[:, 2], label=r'$\theta_z$')
# plt.xlabel('distance traveled [m]')
# plt.ylabel('error [m] or [rad]')
# plt.legend()

# trans_rmse, rot_rmse = tm.rmse()
# plt.figure()
# plt.plot(tm.distances, trans_rmse, label='trans rmse')
# plt.plot(tm.distances, rot_rmse, label='rot rmse')
# plt.xlabel('distance traveled [m]')
# plt.ylabel('rmse [m] or [aa]')
# plt.legend()

trans_armse, rot_armse = tm.armse()
print('trans armse: {} meters'.format(trans_armse))
print('rot armse: {} deg'.format(rot_armse * 180. / np.pi))

plt.show()
