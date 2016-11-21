import numpy as np


class Lidar2D:
    """2D lidar model."""

    def __init__(self):
        pass


class StereoCamera:
    """Pinhole stereo camera model with the origin in left camera."""

    def __init__(self, cu, cv, fu, fv, b):
        self.cu = cu
        self.cv = cv
        self.fu = fu
        self.fv = fv
        self.b = b

    def project(self, pt_c, compute_jacobian=None):
        """Project a 3D point in the camera frame into (u,v,d) coordinates."""
        one_over_z = 1 / pt_c[2]

        obs = np.array([self.fu * pt_c[0] * one_over_z + self.cu,
                        self.fv * pt_c[1] * one_over_z + self.cv,
                        self.fu * self.b * one_over_z])

        if compute_jacobian:
            jacobian = np.empty([3, 3])

            one_over_z2 = one_over_z * one_over_z

            # d(u) / d(pt_c)
            jacobian[0, 0] = self.fu * one_over_z
            jacobian[0, 1] = 0
            jacobian[0, 2] = -self.fu * pt_c[0] * one_over_z2

            # d(v) / d(pt_c)
            jacobian[1, 0] = 0
            jacobian[1, 1] = self.fv * one_over_z
            jacobian[1, 2] = -self.fv * pt_c[1] * one_over_z2

            # d(d) / d(pt_c)
            jacobian[2, 0] = 0
            jacobian[2, 1] = 0
            jacobian[2, 2] = -self.fu * self.b * one_over_z2

            return obs, jacobian

        return obs

    def triangulate(self, obs, compute_jacobian=None):
        """Triangulate a 3D point in the camera frame from (u,v,d) coordinates."""
        b_over_d = self.b / obs[2]
        fu_over_fv = self.fu / self.fv

        pt_c = np.array([(obs[0] - self.cu) * b_over_d,
                         (obs[1] - self.cv) * b_over_d * fu_over_fv,
                         self.fu * b_over_d])

        if compute_jacobian:
            jacobian = np.empty([3, 3])

            b_over_d2 = b_over_d / obs[2]

            # d(x) / d(obs)
            jacobian[0, 0] = b_over_d
            jacobian[0, 1] = 0
            jacobian[0, 2] = (self.cu - obs[0]) * b_over_d2

            # d(y) / d(obs)
            jacobian[1, 0] = 0
            jacobian[1, 1] = b_over_d * fu_over_fv
            jacobian[1, 2] = (self.cv - obs[1]) * b_over_d2 * fu_over_fv

            # d(z) / d(obs)
            jacobian[2, 0] = 0
            jacobian[2, 1] = 0
            jacobian[2, 2] = -self.fu * b_over_d2

            return pt_c, jacobian

        return pt_c

    def __repr__(self):
        return "StereoCamera:\n cu: {:f}\n cv: {:f}\n fu: {:f}\n fv: {:f}\n" \
               "  b: {:f}\n".format(self.cu, self.cv, self.fu, self.fv, self.b)
