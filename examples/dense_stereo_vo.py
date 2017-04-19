import numpy as np
import matplotlib.pyplot as plt

import pykitti

from liegroups import SE3
from pyslam.pipelines import DenseStereoPipeline
from pyslam.sensors import StereoCamera

import time

# Load KITTI data
basedir = '/Users/leeclement/Desktop/KITTI/raw/'
date = '2011_09_26'
drive = '0019'

dataset = pykitti.raw(basedir, date, drive,
                      frames=range(0, 10), imformat='cv2')

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
vo.problem_options.print_summary = True

errs = np.empty([len(dataset), 6])
pos_GT = np.empty([len(dataset), 3])
pos_est = np.empty(pos_GT.shape)
for c_idx, (impair, oxts) in enumerate(zip(dataset.gray, dataset.oxts)):
    print('Image {}'.format(c_idx))

    start = time.perf_counter()

    vo.track(impair[0], impair[1])

    end = time.perf_counter()
    print('Elapsed time: {} s'.format(end - start))

    T_c_w_GT = T_cam0_imu * SE3.from_matrix(oxts.T_w_imu).inv()
    T_c_w_est = vo.T_c_w[c_idx]

    # errs[c_idx, :] = SE3.log(T_c_w_GT * T_c_w_est.inv())
    pos_GT[c_idx, :] = (T_c_w_GT.inv() * np.array([0, 0, 0, 1]))[0:3]
    pos_est[c_idx, :] = (T_c_w_est.inv() * np.array([0, 0, 0, 1]))[0:3]


# plt.plot(errs[:, 0], label='x')
# plt.plot(errs[:, 1], label='y')
# plt.plot(errs[:, 2], label='z')
# plt.plot(errs[:, 3], label=r'$\theta_x$')
# plt.plot(errs[:, 4], label=r'$\theta_y$')
# plt.plot(errs[:, 5], label=r'$\theta_z$')
# plt.legend()
# plt.show()

# plt.plot(pos_GT[:, 0], pos_GT[:, 1], label='GT')
# plt.plot(pos_est[:, 0], pos_est[:, 1], label='est')
# plt.legend()
# plt.show()
