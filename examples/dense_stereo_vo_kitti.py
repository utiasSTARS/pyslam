import numpy as np
import matplotlib.pyplot as plt

import pykitti

from liegroups import SE3
from pyslam.pipelines import DenseStereoPipeline
from pyslam.sensors import StereoCamera
from pyslam import metrics

import time

# Load KITTI data
basedir = '/Users/leeclement/Desktop/KITTI/raw/'
date = '2011_09_30'
drive = '0034'
frames = None

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

# Pipeline
vo = DenseStereoPipeline(camera, first_pose=T_0_w)
for c_idx, impair in enumerate(dataset.gray):
    print('Image {}'.format(c_idx))

    vo.track(impair[0], impair[1])

# Compute and plot errors
errs = np.empty([len(dataset), 6])
pos_GT = np.empty([len(dataset), 3])
pos_est = np.empty(pos_GT.shape)

for c_idx, (T_c_w_est, oxts) in enumerate(zip(vo.T_c_w, dataset.oxts)):
    T_c_w_GT = T_cam0_imu * SE3.from_matrix(oxts.T_w_imu).inv()

    errs[c_idx, :] = SE3.log(T_c_w_GT * T_c_w_est.inv())
    # rotate translational errors into world frame
    errs[c_idx, 0:3] = T_c_w_GT.inv().rot * errs[c_idx, 0:3]
    pos_GT[c_idx, :] = T_c_w_GT.inv().trans
    pos_est[c_idx, :] = T_c_w_est.inv().trans

dist_GT = np.diff(pos_GT[:, 0:3], axis=0)
dist_GT = np.cumsum(np.linalg.norm(dist_GT, axis=1))
dist_GT = np.append([0.], dist_GT)

plt.figure()
plt.plot(dist_GT, errs[:, 0], label='x')
plt.plot(dist_GT, errs[:, 1], label='y')
plt.plot(dist_GT, errs[:, 2], label='z')
plt.plot(dist_GT, errs[:, 3], label=r'$\theta_x$')
plt.plot(dist_GT, errs[:, 4], label=r'$\theta_y$')
plt.plot(dist_GT, errs[:, 5], label=r'$\theta_z$')
plt.xlabel('distance traveled [m]')
plt.ylabel('error [m] or [rad]')
plt.legend()

# plt.figure()
# plt.plot(dist_GT, pos_est[:, 0] - pos_GT[:, 0], label='x')
# plt.plot(dist_GT, pos_est[:, 1] - pos_GT[:, 1], label='y')
# plt.plot(dist_GT, pos_est[:, 2] - pos_GT[:, 2], label='z')
# plt.legend()

plt.figure()
plt.plot(dist_GT, metrics.rmse(errs[:, 0:3]), label='trans rmse')
# plt.plot(dist_GT, metrics.rmse(errs[:, 3:6]), label='rot rmse')
plt.xlabel('distance traveled [m]')
plt.ylabel('rmse [m] or [aa]')
plt.legend()
print('trans armse: {}'.format(metrics.armse(errs[:, 0:3])))
print('rot armse: {}'.format(metrics.armse(errs[:, 3:6])))

plt.figure()
plt.plot(pos_GT[:, 0], pos_GT[:, 1], label='GT')
plt.plot(pos_est[:, 0], pos_est[:, 1], label='est')
plt.legend()

plt.show()
