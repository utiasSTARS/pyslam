import numpy as np

from liegroups import *


class PoseCost:
    """Unary pose cost given absolute pose measurement in SE(2) or SE(3)."""

    def __init__(self, T_obs, weight):
        self.T_obs = T_obs
        self.weight = weight
        self.obstype = type(T_obs)

    def evaluate(self, params, compute_jacobians=None):
        T_est = params[0]

        residual = self.obstype.log(T_est * self.T_obs.inv())

        if compute_jacobians:
            jacobians = [None for _ in range(len(params))]

            if compute_jacobians[0]:
                jacobians[0] = np.identity(self.obstype.dof)

            return residual, jacobians

        return residual


class PoseToPoseCost:
    """Binary pose-to-pose cost given relative pose mesurement in SE(2) or SE(3)."""

    def __init__(self, T_2_1_obs, weight):
        self.T_2_1_obs = T_2_1_obs
        self.weight = weight
        self.obstype = type(T_2_1_obs)

    def evaluate(self, params, compute_jacobians=None):
        T_1_0_est = params[0]
        T_2_0_est = params[1]

        residual = self.obstype.log(
            T_2_0_est * T_1_0_est.inv() * self.T_2_1_obs.inv())

        if compute_jacobians:
            jacobians = [None for _ in range(len(params))]

            if compute_jacobians[0]:
                jacobians[0] = -T_2_0_est.adjoint()

            if compute_jacobians[1]:
                jacobians[1] = np.identity(self.obstype.dof)

            return residual, jacobians

        return residual


class ReprojectionCost:
    """Reprojection error for any kind of camera."""

    def __init__(self, camera, obs, weight):
        self.camera = camera
        self.obs = obs
        self.weight = weight

    def evaluate(self, params, compute_jacobians=None):
        T_cam_w = params[0]
        pt_w = params[1]
        pt_cam = T_cam_w * pt_w

        if compute_jacobians:
            jacobians = [None for _ in range(len(params))]

            if any(compute_jacobians):
                predicted_obs, cam_jacobian = self.camera.project(pt_cam, True)
                residual = predicted_obs - self.obs

            if compute_jacobians[0]:
                jacobians[0] = cam_jacobian.dot(SE3.odot(pt_cam))

            if compute_jacobians[1]:
                jacobians[1] = cam_jacobian.dot(T_cam_w.rot.as_matrix())

            return residual, jacobians

        residual = self.camera.project(pt_cam) - self.obs
        return residual
