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

    def _valid_mask(self, uvd):
        return (uvd[:, 2] > 0.) & (uvd[:, 2] < self.w) & \
            (uvd[:, 1] > 0.) & (uvd[:, 1] < self.h) & \
            (uvd[:, 0] > 0.) & (uvd[:, 0] < self.w)

    def is_valid_measurement(self, uvd):
        """Check if one or more uvd measurements is valid.
           Returns boolean mask of valid measurements.
        """
        uvd = np.atleast_2d(uvd)

        if not uvd.shape[1] == 3:
            raise ValueError("uvd must have shape (3,) or (N,3)")

        return np.squeeze(self._valid_mask(uvd))

    def _project(self, pt_c, out):
        one_over_z = 1. / pt_c[:, 2]
        out[:, 0] = self.fu * pt_c[:, 0] * one_over_z + self.cu
        out[:, 1] = self.fv * pt_c[:, 1] * one_over_z + self.cv
        out[:, 2] = self.fu * self.b * one_over_z

    def _project_jacobian(self, pt_c, out):
        one_over_z = 1. / pt_c[:, 2]
        one_over_z2 = one_over_z * one_over_z

        # d(u) / d(pt_c)
        out[:, 0, 0] = self.fu * one_over_z
        out[:, 0, 1] = 0.
        out[:, 0, 2] = -self.fu * pt_c[:, 0] * one_over_z2

        # d(v) / d(pt_c)
        out[:, 1, 0] = 0.
        out[:, 1, 1] = self.fv * one_over_z
        out[:, 1, 2] = -self.fv * pt_c[:, 1] * one_over_z2

        # d(d) / d(pt_c)
        out[:, 2, 0] = 0.
        out[:, 2, 1] = 0.
        out[:, 2, 2] = -self.fu * self.b * one_over_z2

    def project(self, pt_c, compute_jacobians=None):
        """Project 3D point(s) in the sensor frame into (u,v,d) coordinates."""
        pt_c = np.atleast_2d(pt_c)

        if not pt_c.shape[1] == 3:
            raise ValueError("pt_c must have shape (3,) or (N,3)")

        uvd = np.empty(pt_c.shape)
        self._project(pt_c, uvd)

        if compute_jacobians:
            jacobians = np.empty([pt_c.shape[0], 3, 3])
            self._project_jacobian(pt_c, jacobians)

            return np.squeeze(uvd), np.squeeze(jacobians)

        return np.squeeze(uvd)

    def _triangulate(self, uvd, out):
        b_over_d = self.b / uvd[:, 2]
        fu_over_fv = self.fu / self.fv

        out[:, 0] = (uvd[:, 0] - self.cu) * b_over_d
        out[:, 1] = (uvd[:, 1] - self.cv) * b_over_d * fu_over_fv
        out[:, 2] = self.fu * b_over_d

    def _triangulate_jacobian(self, uvd, out):
        b_over_d = self.b / uvd[:, 2]
        b_over_d2 = b_over_d / uvd[:, 2]
        fu_over_fv = self.fu / self.fv

        # d(x) / d(uvd)
        out[:, 0, 0] = b_over_d
        out[:, 0, 1] = 0.
        out[:, 0, 2] = (self.cu - uvd[:, 0]) * b_over_d2

        # d(y) / d(uvd)
        out[:, 1, 0] = 0.
        out[:, 1, 1] = b_over_d * fu_over_fv
        out[:, 1, 2] = (self.cv - uvd[:, 1]) * b_over_d2 * fu_over_fv

        # d(z) / d(uvd)
        out[:, 2, 0] = 0.
        out[:, 2, 1] = 0.
        out[:, 2, 2] = -self.fu * b_over_d2

    def triangulate(self, uvd, compute_jacobians=None):
        """Triangulate 3D point(s) in the sensor frame from (u,v,d)."""
        uvd = np.atleast_2d(uvd)

        if not uvd.shape[1] == 3:
            raise ValueError("uvd must have shape (3,) or (N,3)")

        pt_c = np.empty(uvd.shape)
        self._triangulate(uvd, pt_c)

        if compute_jacobians:
            jacobians = np.empty([uvd.shape[0], 3, 3])
            self._triangulate_jacobian(uvd, jacobians)

            return np.squeeze(pt_c), np.squeeze(jacobians)

        return np.squeeze(pt_c)

    def __repr__(self):
        return "{}:\n| cu: {:f}\n| cv: {:f}\n| fu: {:f}\n| fv: {:f}\n|" \
               "  b: {:f}\n|  w: {:d}\n|  h: {:d}\n".format(
                   self.__class__.__name__, self.cu, self.cv,
                   self.fu, self.fv,
                   self.b,
                   self.w, self.h)


class StereoCameraTorch(StereoCamera):
    """Torch specialization of StereoCamera."""

    def is_valid_measurement(self, uvd):
        if uvd.dim() < 2:
            uvd = uvd.unsqueeze(dim=0)

        if not uvd.shape[1] == 3:
            raise ValueError("uvd must have shape (3,) or (N,3)")

        return self._valid_mask(uvd).squeeze_()

    def project(self, pt_c, compute_jacobians=None):
        if pt_c.dim() < 2:
            pt_c = pt_c.unsqueeze(dim=0)

        if not pt_c.shape[1] == 3:
            raise ValueError("pt_c must have shape (3,) or (N,3)")

        uvd = pt_c.__class__(pt_c.shape)
        self._project(pt_c, uvd)

        if compute_jacobians:
            jacobians = pt_c.__class__(pt_c.shape[0], 3, 3)
            self._project_jacobian(uvd, jacobians)

            return uvd.squeeze_(), jacobians.squeeze_()

        return uvd.squeeze_()

    def triangulate(self, uvd, compute_jacobians=None):
        if uvd.dim() < 2:
            uvd = uvd.unsqueeze(dim=0)

        if not uvd.shape[1] == 3:
            raise ValueError("uvd must have shape (3,) or (N,3)")

        pt_c = uvd.__class__(uvd.shape)
        self._triangulate(uvd, pt_c)

        if compute_jacobians:
            jacobians = uvd.__class__(uvd.shape[0], 3, 3)
            self._triangulate_jacobian(pt_c, jacobians)

            return pt_c.squeeze_(), jacobians.squeeze_()

        return pt_c.squeeze_()
