import numpy as np
import matplotlib.pyplot as plt

import cv2
import pykitti

from liegroups import SE3
from pyslam.problem import Options, Problem
from pyslam.sensors import StereoCamera
from pyslam.costs import PhotometricCost
from pyslam.utils import bilinear_interpolate

# Load KITTI data
basedir = '/Users/leeclement/Desktop/odometry_raw/'
date = '2011_09_30'
drive = '0018'
frame_range = [0, 1]

dataset = pykitti.raw(basedir, date, drive, frame_range)
dataset.load_calib()
dataset.load_gray(format='cv2')
dataset.load_oxts()

T_0_w = SE3.from_matrix(dataset.calib.T_cam0_imu.dot(
    np.linalg.inv(dataset.oxts[0].T_w_imu)))
T_1_w = SE3.from_matrix(dataset.calib.T_cam0_imu.dot(
    np.linalg.inv(dataset.oxts[1].T_w_imu)))
T_1_0 = T_1_w * T_0_w.inv()
# T_1_0 = SE3.identity()

# Disparity computation parameters
window_size = 15
min_disp = 0
max_disp = 64 + min_disp

# Use semi-global block matching
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=max_disp - min_disp,
    blockSize=window_size)

# Use regular block matching
# stereo = cv2.StereoBM_create(
#     numDisparities=max_disp - min_disp, blockSize=window_size)

disp = []
for impair in dataset.gray:
    # imL = cv2.pyrDown(impair.left[1:, :])
    # imR = cv2.pyrDown(impair.right[1:, :])
    # disp.append(2. * cv2.pyrUp(stereo.compute(imL, imR)))

    imL = impair.left
    imR = impair.right
    disp.append(stereo.compute(imL[1:, :], imR[1:, :]))

disp = [np.float32(d) / 16. for d in disp]
for i, d in enumerate(disp):
    disp[i][d < min_disp + 1] = np.nan
    disp[i][d > max_disp] = np.nan
    missing_row = np.empty([1, d.shape[1]])
    missing_row.fill(np.nan)
    disp[i] = np.vstack([disp[i], missing_row])


# Create the camera
fu = dataset.calib.K_cam0[0, 0]
fv = dataset.calib.K_cam0[1, 1]
cu = dataset.calib.K_cam0[0, 2]
cv = dataset.calib.K_cam0[1, 2]
b = dataset.calib.b_gray
w = dataset.gray[0].left.shape[1]
h = dataset.gray[0].left.shape[0]
camera = StereoCamera(cu, cv, fu, fv, b, w, h)

# Create the cost function
cost = PhotometricCost(camera, dataset.gray[0].left, disp[
                       0], dataset.gray[1].left, [], 1.)

residual = cost.evaluate([T_1_0])
