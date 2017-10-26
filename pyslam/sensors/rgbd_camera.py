import numpy as np


class RGBDCamera:
    """Pinhole RGB-D camera model."""

    def __init__(self, cu, cv, fu, fv, w, h):
        self.cu = float(cu)
        self.cv = float(cv)
        self.fu = float(fu)
        self.fv = float(fv)
        self.w = int(w)
        self.h = int(h)

    def _valid_mask(self, uvz):
        return (uvz[:, 2] > 0.) & \
            (uvz[:, 1] > 0.) & (uvz[:, 1] < self.h) & \
            (uvz[:, 0] > 0.) & (uvz[:, 0] < self.w)

    def is_valid_measurement(self, uvz):
        """Check if one or more uvz measurements is valid.
           Returns boolean mask of valid measurements.
        """
        uvz = np.atleast_2d(uvz)

        if not uvz.shape[1] == 3:
            raise ValueError("uvz must have shape (3,) or (N,3)")

        return np.squeeze(self._valid_mask(uvz))

    def _project(self, pt_c, out):
        one_over_z = 1. / pt_c[:, 2]
        out[:, 0] = self.fu * pt_c[:, 0] * one_over_z + self.cu
        out[:, 1] = self.fv * pt_c[:, 1] * one_over_z + self.cv
        out[:, 2] = pt_c[:, 2]

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
        out[:, 2, 2] = 1.

    def project(self, pt_c, compute_jacobians=None):
        """Project 3D point(s) in the sensor frame into (u,v,z) coordinates."""
        # Convert to 2D array if it's just a single point.
        # We'll remove any singleton dimensions at the end.
        pt_c = np.atleast_2d(pt_c)

        if not pt_c.shape[1] == 3:
            raise ValueError("pt_c must have shape (3,) or (N,3)")

        # Now do the actual math
        uvz = np.empty(pt_c.shape)
        self._project(pt_c, uvz)

        if compute_jacobians:
            jacobians = np.empty([pt_c.shape[0], 3, 3])
            self._project_jacobian(pt_c, jacobians)

            return np.squeeze(uvz), np.squeeze(jacobians)

        return np.squeeze(uvz)

    def _triangulate(self, uvz, out):
        out[:, 0] = (uvz[:, 0] - self.cu) * uvz[:, 2] / self.fu
        out[:, 1] = (uvz[:, 1] - self.cv) * uvz[:, 2] / self.fv
        out[:, 2] = uvz[:, 2]

    def _triangulate_jacobian(self, uvz, out):
        one_over_fu = 1. / self.fu
        one_over_fv = 1. / self.fv

        # d(x) / d(uvz)
        out[:, 0, 0] = uvz[:, 2] * one_over_fu
        out[:, 0, 1] = 0.
        out[:, 0, 2] = (uvz[:, 0] - self.cu) * one_over_fu

        # d(y) / d(uvz)
        out[:, 1, 0] = 0.
        out[:, 1, 1] = uvz[:, 2] * one_over_fv
        out[:, 1, 2] = (uvz[:, 1] - self.cv) * one_over_fv

        # d(z) / d(uvz)
        out[:, 2, 0] = 0.
        out[:, 2, 1] = 0.
        out[:, 2, 2] = 1.

    def triangulate(self, uvz, compute_jacobians=None):
        """Triangulate 3D point(s) in the sensor frame from (u,v,z)."""
        # Convert to 2D array if it's just a single point
        # We'll remove any singleton dimensions at the end.
        uvz = np.atleast_2d(uvz)

        if not uvz.shape[1] == 3:
            raise ValueError("uvz must have shape (3,) or (N,3)")

        # Now do the actual math
        pt_c = np.empty(uvz.shape)
        self._triangulate(uvz, pt_c)

        if compute_jacobians:
            jacobians = np.empty([uvz.shape[0], 3, 3])
            self._triangulate_jacobian(uvz, jacobians)

            return np.squeeze(pt_c), np.squeeze(jacobians)

        return np.squeeze(pt_c)

    def __repr__(self):
        return "{}:\n| cu: {:f}\n| cv: {:f}\n| fu: {:f}\n| fv: {:f}\n|" \
               " w: {:d}\n|  h: {:d}\n".format(
                   self.__class__.__name__,
                   self.cu, self.cv,
                   self.fu, self.fv,
                   self.w, self.h)


class RGBDCameraTorch(RGBDCamera):
    """Torch specialization of RGBDCamera."""

    def is_valid_measurement(self, uvz):
        if uvz.dim() < 2:
            uvz = uvz.unsqueeze(dim=0)

        if not uvz.shape[1] == 3:
            raise ValueError("uvz must have shape (3,) or (N,3)")

        return self._valid_mask(uvz).squeeze_()

    def project(self, pt_c, compute_jacobians=None):
        if pt_c.dim() < 2:
            pt_c = pt_c.unsqueeze(dim=0)

        if not pt_c.shape[1] == 3:
            raise ValueError("pt_c must have shape (3,) or (N,3)")

        uvz = pt_c.__class__(pt_c.shape)
        self._project(pt_c, uvz)

        if compute_jacobians:
            jacobians = pt_c.__class__(pt_c.shape[0], 3, 3)
            self._project_jacobian(uvz, jacobians)

            return uvz.squeeze_(), jacobians.squeeze_()

        return uvz.squeeze_()

    def triangulate(self, uvz, compute_jacobians=None):
        if uvz.dim() < 2:
            uvz = uvz.unsqueeze(dim=0)

        if not uvz.shape[1] == 3:
            raise ValueError("uvz must have shape (3,) or (N,3)")

        pt_c = uvz.__class__(uvz.shape)
        self._triangulate(uvz, pt_c)

        if compute_jacobians:
            jacobians = uvz.__class__(uvz.shape[0], 3, 3)
            self._triangulate_jacobian(pt_c, jacobians)

            return pt_c.squeeze_(), jacobians.squeeze_()

        return pt_c.squeeze_()
