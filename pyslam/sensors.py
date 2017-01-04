import numpy as np


class StereoCamera:
    """Pinhole stereo camera model with the origin in left camera."""

    def __init__(self, cu, cv, fu, fv, b, w, h):
        self.cu = cu
        self.cv = cv
        self.fu = fu
        self.fv = fv
        self.b = b
        self.w = w
        self.h = h

    def project(self, pt_c, compute_jacobians=None):
        """Project a 3D point in the sensor frame into (u,v,d) coordinates."""
        one_over_z = 1 / pt_c[2]

        uvd = np.array([self.fu * pt_c[0] * one_over_z + self.cu,
                        self.fv * pt_c[1] * one_over_z + self.cv,
                        self.fu * self.b * one_over_z])

        if self.is_valid_measurement(uvd):
            if compute_jacobians:
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

                return uvd, jacobian

            return uvd

        raise ValueError('Got an invalid measurement in StereoCamera.project')

    def triangulate(self, uvd, compute_jacobians=None):
        """Triangulate a 3D point in the sensor frame from (u,v,d)."""
        if self.is_valid_measurement(uvd):
            b_over_d = self.b / uvd[2]
            fu_over_fv = self.fu / self.fv

            pt_c = np.array([(uvd[0] - self.cu) * b_over_d,
                             (uvd[1] - self.cv) * b_over_d * fu_over_fv,
                             self.fu * b_over_d])

            if compute_jacobians:
                jacobian = np.empty([3, 3])

                b_over_d2 = b_over_d / uvd[2]

                # d(x) / d(uvd)
                jacobian[0, 0] = b_over_d
                jacobian[0, 1] = 0
                jacobian[0, 2] = (self.cu - uvd[0]) * b_over_d2

                # d(y) / d(uvd)
                jacobian[1, 0] = 0
                jacobian[1, 1] = b_over_d * fu_over_fv
                jacobian[1, 2] = (self.cv - uvd[1]) * b_over_d2 * fu_over_fv

                # d(z) / d(uvd)
                jacobian[2, 0] = 0
                jacobian[2, 1] = 0
                jacobian[2, 2] = -self.fu * b_over_d2

                return pt_c, jacobian

            return pt_c

        raise ValueError(
            'Got an invalid measurement in StereoCamera.triangulate')

    def is_valid_measurement(self, uvd):
        return uvd[0] >= 0 and uvd[0] < self.w and \
            uvd[1] >= 0 and uvd[1] < self.h and \
            uvd[2] > 0

    def __repr__(self):
        return "StereoCamera:\n cu: {:f}\n cv: {:f}\n fu: {:f}\n fv: {:f}\n" \
               "  b: {:f}\n  w: {:d}\n  h: {:d}\n".format(self.cu, self.cv,
                                                          self.fu, self.fv,
                                                          self.b,
                                                          self.w, self.h)
