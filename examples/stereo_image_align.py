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

# Disparity computation parameters
window_size = 5
min_disp = 0
max_disp = 32 + min_disp

# Use semi-global block matching
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=max_disp - min_disp,
    blockSize=window_size)

# Use regular block matching
# stereo = cv2.StereoBM_create(
#     numDisparities=max_disp - min_disp, blockSize=window_size)

impairs = []
for ims in dataset.gray:
    imL = cv2.pyrDown(ims.left)
    imR = cv2.pyrDown(ims.right)
    impairs.append([imL, imR])

disp = []
for ims in impairs:
    d = stereo.compute(ims[0], ims[1])
    disp.append(d.astype(float) / 16.)

for i, d in enumerate(disp):
    disp[i][d < min_disp + 1] = np.nan
    disp[i][d > max_disp] = np.nan
    # missing_row = np.empty([1, d.shape[1]])
    # missing_row.fill(np.nan)
    # disp[i] = np.vstack([disp[i], missing_row])

# Compute image jacobians
im_jac = []
for ims in impairs:
    # gradx = cv2.Sobel(ims[0], -1, 1, 0)
    # grady = cv2.Sobel(ims[0], -1, 0, 1)
    gradx = cv2.Scharr(ims[0], -1, 1, 0)
    grady = cv2.Scharr(ims[0], -1, 0, 1)
    im_jac.append(np.array([gradx.astype(float) / 255.,
                            grady.astype(float) / 255.]))

# Create the camera
fu = dataset.calib.K_cam0[0, 0] / 2.
fv = dataset.calib.K_cam0[1, 1] / 2.
cu = dataset.calib.K_cam0[0, 2] / 2.
cv = dataset.calib.K_cam0[1, 2] / 2.
b = dataset.calib.b_gray
w = dataset.gray[0].left.shape[1] / 2.
h = dataset.gray[0].left.shape[0] / 2.
camera = StereoCamera(cu, cv, fu, fv, b, w, h)

# Create the cost function
im_ref = impairs[0][0].astype(float) / 255.
disp_ref = disp[0]
im_track = impairs[1][0].astype(float) / 255.
jac_ref = im_jac[0]
cost = PhotometricCost(camera, im_ref, disp_ref, jac_ref, im_track, 1.)

T_0_w = SE3.from_matrix(dataset.calib.T_cam0_imu.dot(
    np.linalg.inv(dataset.oxts[0].T_w_imu)))
T_1_w = SE3.from_matrix(dataset.calib.T_cam0_imu.dot(
    np.linalg.inv(dataset.oxts[1].T_w_imu)))
T_0_w.normalize()
T_1_w.normalize()

params_init = {'T_1_0': T_1_w * T_0_w.inv()}
# params_init = {'T_1_0': SE3.identity()}

# residual, jacobians = cost.evaluate([params_init['T_1_0']], [True])
# residual = cost.evaluate([params_init['T_1_0']])

# Optimize
options = Options()
options.allow_nondecreasing_steps = True
options.max_nondecreasing_steps = 3

problem = Problem(options)
problem.add_residual_block(cost, ['T_1_0'])
problem.initialize_params(params_init)
params_final = problem.solve()
