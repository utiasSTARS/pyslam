import numpy as np
import matplotlib.pyplot as plt

import pykitti

from liegroups import SE3
from pyslam.pipelines import DenseStereoPipeline
from pyslam.sensors import StereoCamera

import time

# Load KITTI data
basedir = '/Users/leeclement/Desktop/KITTI/raw/'
date = '2011_09_30'
drive = '0018'
frame_range = range(0, 50)

dataset = pykitti.raw(basedir, date, drive, frame_range)
dataset.load_calib()
dataset.load_gray(format='cv2')
dataset.load_oxts()

# Parameters to estimate
T_cam0_imu = SE3.from_matrix(dataset.calib.T_cam0_imu)
T_0_w = T_cam0_imu * \
    SE3.from_matrix(dataset.oxts[0].T_w_imu).inv()

# Create the camera
fu = dataset.calib.K_cam0[0, 0]
fv = dataset.calib.K_cam0[1, 1]
cu = dataset.calib.K_cam0[0, 2]
cv = dataset.calib.K_cam0[1, 2]
b = dataset.calib.b_gray
w = dataset.gray[0].left.shape[1]
h = dataset.gray[0].left.shape[0]
camera = StereoCamera(cu, cv, fu, fv, b, w, h)

# Pipeline
vo = DenseStereoPipeline(camera, first_pose=T_0_w)
vo.problem_options.print_summary = False

for c_idx, f_idx in enumerate(frame_range):
    start = time.perf_counter()

    vo.track(dataset.gray[c_idx].left, dataset.gray[c_idx].right)

    end = time.perf_counter()
    print('Elapsed time: {} s'.format(end - start))

    T_c_w_GT = T_cam0_imu * SE3.from_matrix(dataset.oxts[c_idx].T_w_imu).inv()
    T_c_w_est = vo.keyframes[c_idx].T_c_w
    print('Error in T_{}_w: {}'.format(f_idx,
                                       SE3.log(T_c_w_GT * T_c_w_est.inv())))
