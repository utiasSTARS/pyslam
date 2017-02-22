import numpy as np
import matplotlib.pyplot as plt

import cv2
import pykitti

from liegroups import SE3
from pyslam.problem import Options, Problem
from pyslam.sensors import StereoCamera
from pyslam.costs import PhotometricCost
import time

# Load KITTI data
basedir = '/Users/leeclement/Desktop/KITTI/raw/'
date = '2011_09_30'
drive = '0018'
frame_range = [0, 1]

dataset = pykitti.raw(basedir, date, drive, frame_range)
dataset.load_calib()
dataset.load_gray(format='cv2')
dataset.load_oxts()

# Parameters to estimate
T_0_w = SE3.from_matrix(dataset.calib.T_cam0_imu.dot(
    np.linalg.inv(dataset.oxts[0].T_w_imu)))
T_1_w = SE3.from_matrix(dataset.calib.T_cam0_imu.dot(
    np.linalg.inv(dataset.oxts[1].T_w_imu)))
T_0_w.normalize()
T_1_w.normalize()
T_1_0_true = T_1_w * T_0_w.inv()

# params_init = {'T_1_0': T_1_0_true}
params_init = {'T_1_0': SE3.identity()}

# Scaling parameters
pyrlevels = [3, 2, 1]

params = params_init

options = Options()
options.allow_nondecreasing_steps = True
options.max_nondecreasing_steps = 3
options.min_cost_decrease = 0.99
# options.max_iters = 100
# options.print_iter_summary = True

for pyrlevel in pyrlevels:
    pyrfactor = 1. / 2.**pyrlevel

    # Disparity computation parameters
    window_size = 5
    min_disp = 1
    max_disp = np.max([16, np.int(64 * pyrfactor)]) + min_disp

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
        imL = ims.left
        imR = ims.right
        for _ in range(pyrlevel):
            imL = cv2.pyrDown(imL)
            imR = cv2.pyrDown(imR)
        impairs.append([imL, imR])

    disp = []
    for ims in impairs:
        d = stereo.compute(ims[0], ims[1])
        disp.append(d.astype(float) / 16.)

    for i, d in enumerate(disp):
        disp[i][d < min_disp + 1] = np.nan
        disp[i][d > max_disp] = np.nan

    # Compute image jacobians
    im_jac = []
    for ims in impairs:
        gradx = cv2.Sobel(ims[0], -1, 1, 0)
        grady = cv2.Sobel(ims[0], -1, 0, 1)
        # gradx = cv2.Scharr(ims[0], -1, 1, 0)
        # grady = cv2.Scharr(ims[0], -1, 0, 1)
        im_jac.append(np.array([gradx.astype(float) / 255.,
                                grady.astype(float) / 255.]))

    # Create the camera
    fu = dataset.calib.K_cam0[0, 0] * pyrfactor
    fv = dataset.calib.K_cam0[1, 1] * pyrfactor
    cu = dataset.calib.K_cam0[0, 2] * pyrfactor
    cv = dataset.calib.K_cam0[1, 2] * pyrfactor
    b = dataset.calib.b_gray
    w = impairs[0][0].shape[1]
    h = impairs[0][0].shape[0]
    camera = StereoCamera(cu, cv, fu, fv, b, w, h)

    # Create the cost function
    im_ref = impairs[0][0].astype(float) / 255.
    disp_ref = disp[0]
    im_track = impairs[1][0].astype(float) / 255.
    jac_ref = im_jac[0]
    cost = PhotometricCost(camera, im_ref, disp_ref, jac_ref, im_track, 1.)

    # Timing debug
    # niters = 100
    # start = time.perf_counter()
    # for _ in range(niters):
    #     cost.evaluate([params_init['T_1_0']])
    # end = time.perf_counter()
    # print('cost.evaluate avg {} s', (end - start) / niters)

    # start = time.perf_counter()
    # for _ in range(niters):
    #     cost.evaluate([params_init['T_1_0']], [True])
    # end = time.perf_counter()
    # print('cost.evaluate w jac avg {} s', (end - start) / niters)

    # Optimize
    start = time.perf_counter()

    problem = Problem(options)
    problem.add_residual_block(cost, ['T_1_0'])
    problem.initialize_params(params)
    params = problem.solve()

    end = time.perf_counter()
    print('Elapsed time: {} s'.format(end - start))

    print('Error in T_1_0: {}'.format(
        SE3.log(T_1_0_true * params['T_1_0'].inv())))
