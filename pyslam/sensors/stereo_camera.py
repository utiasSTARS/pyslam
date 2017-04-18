import numpy as np


class StereoCamera:
    """Pinhole stereo camera model with the origin in left camera."""

    def __init__(self, cu, cv, fu, fv, b, w, h):
        self.cu = float(cu)
        self.cv = float(cv)
        self.fu = float(fu)
        self.fv = float(fv)
        self.b = float(b)
        self.w = int(w)
        self.h = int(h)

    def is_valid_measurement(self, uvd):
        """Check if one or more uvd measurements is valid.
           Returns a boolean mask of valid measurements.
        """
        uvd = np.atleast_2d(uvd)

        if not uvd.shape[1] == 3:
            raise ValueError("uvd must have shape (3,) or (N,3)")

        # Do a np.isfinite check, otherwise vectorized > and < print
        # annoying warnings if given nan or inf
        isvalid = np.zeros(uvd.shape[0], dtype=bool)  # fill with false
        isvalid[np.isfinite(uvd).all(axis=1)] = True
        isvalid[isvalid] = (uvd[isvalid, 0] > 0.) & (uvd[isvalid, 0] < self.w) & \
            (uvd[isvalid, 1] > 0.) & (uvd[isvalid, 1] < self.h) & \
            (uvd[isvalid, 2] > 0.)

        return isvalid

    def project(self, pt_c, compute_jacobians=None):
        """Project 3D point(s) in the sensor frame into (u,v,d) coordinates."""
        # Convert to 2D array if it's just a single point.
        # We'll remove any singleton dimensions at the end.
        pt_c = np.atleast_2d(pt_c)

        if not pt_c.shape[1] == 3:
            raise ValueError("pt_c must have shape (3,) or (N,3)")

        # Now do the actual math
        one_over_z = 1 / pt_c[:, 2]

        uvd = np.array([self.fu * pt_c[:, 0] * one_over_z + self.cu,
                        self.fv * pt_c[:, 1] * one_over_z + self.cv,
                        self.fu * self.b * one_over_z]).T

        if compute_jacobians:
            jacobians = np.empty([pt_c.shape[0], 3, 3])

            one_over_z2 = one_over_z * one_over_z

            # d(u) / d(pt_c)
            jacobians[:, 0, 0] = self.fu * one_over_z
            jacobians[:, 0, 1] = 0.
            jacobians[:, 0, 2] = -self.fu * \
                pt_c[:, 0] * one_over_z2

            # d(v) / d(pt_c)
            jacobians[:, 1, 0] = 0.
            jacobians[:, 1, 1] = self.fv * one_over_z
            jacobians[:, 1, 2] = -self.fv * \
                pt_c[:, 1] * one_over_z2

            # d(d) / d(pt_c)
            jacobians[:, 2, 0] = 0.
            jacobians[:, 2, 1] = 0.
            jacobians[:, 2, 2] = -self.fu * self.b * one_over_z2

            return np.squeeze(uvd), np.squeeze(jacobians)

        return np.squeeze(uvd)

    def triangulate(self, uvd, compute_jacobians=None):
        """Triangulate 3D point(s) in the sensor frame from (u,v,d)."""
        # Convert to 2D array if it's just a single point
        # We'll remove any singleton dimensions at the end.
        uvd = np.atleast_2d(uvd)

        if not uvd.shape[1] == 3:
            raise ValueError("uvd must have shape (3,) or (N,3)")

        # Now do the actual math
        b_over_d = self.b / uvd[:, 2]
        fu_over_fv = self.fu / self.fv

        pt_c = np.array([(uvd[:, 0] - self.cu) * b_over_d,
                         (uvd[:, 1] - self.cv) * b_over_d * fu_over_fv,
                         self.fu * b_over_d]).T

        if compute_jacobians:
            jacobians = np.empty([uvd.shape[0], 3, 3])

            b_over_d2 = b_over_d / uvd[:, 2]

            # d(x) / d(uvd)
            jacobians[:, 0, 0] = b_over_d
            jacobians[:, 0, 1] = 0.
            jacobians[:, 0, 2] = (
                self.cu - uvd[:, 0]) * b_over_d2

            # d(y) / d(uvd)
            jacobians[:, 1, 0] = 0.
            jacobians[:, 1, 1] = b_over_d * fu_over_fv
            jacobians[:, 1, 2] = (
                self.cv - uvd[:, 1]) * b_over_d2 * fu_over_fv

            # d(z) / d(uvd)
            jacobians[:, 2, 0] = 0.
            jacobians[:, 2, 1] = 0.
            jacobians[:, 2, 2] = -self.fu * b_over_d2

            return np.squeeze(pt_c), np.squeeze(jacobians)

        return np.squeeze(pt_c)

    def __repr__(self):
        return "StereoCamera:\n cu: {:f}\n cv: {:f}\n fu: {:f}\n fv: {:f}\n" \
               "  b: {:f}\n  w: {:d}\n  h: {:d}\n".format(self.cu, self.cv,
                                                          self.fu, self.fv,
                                                          self.b,
                                                          self.w, self.h)
