import numpy as np
from numba import guvectorize, float32, float64, boolean

NUMBA_COMPILATION_TARGET = 'parallel'


class RGBDCamera:
    """Pinhole RGB-D camera model."""

    def __init__(self, cu, cv, fu, fv, w, h):
        self.cu = float(cu)
        self.cv = float(cv)
        self.fu = float(fu)
        self.fv = float(fv)
        self.w = int(w)
        self.h = int(h)

    def clone(self):
        return self.__class__(self.cu, self.cv,
                              self.fu, self.fv,
                              self.w, self.h)

    def compute_pixel_grid(self):
        self.u_grid, self.v_grid = np.meshgrid(range(0, self.w),
                                               range(0, self.h), indexing='xy')
        self.u_grid = self.u_grid.astype(float)
        self.v_grid = self.v_grid.astype(float)

    def is_valid_measurement(self, uvz):
        """Check if one or more uvz measurements is valid.
           Returns indices of valid measurements.
        """
        uvz = np.atleast_2d(uvz)

        if not uvz.shape[1] == 3:
            raise ValueError("uvz must have shape (3,) or (N,3)")

        return _rgbd_validate(uvz, [self.w, self.h])

    def project(self, pt_c, compute_jacobians=None):
        """Project 3D point(s) in the sensor frame into (u,v,d) coordinates."""
        # Convert to 2D array if it's just a single point.
        # We'll remove any singleton dimensions at the end.
        pt_c = np.atleast_2d(pt_c)

        if not pt_c.shape[1] == 3:
            raise ValueError("pt_c must have shape (3,) or (N,3)")

        # Now do the actual math
        params = self.cu, self.cv, self.fu, self.fv
        uvz = _rgbd_project(pt_c, params)

        if compute_jacobians:
            jacobians = _rgbd_project_jacobian(pt_c, params)

            return np.squeeze(uvz), np.squeeze(jacobians)

        return np.squeeze(uvz)

    def triangulate(self, uvz, compute_jacobians=None):
        """Triangulate 3D point(s) in the sensor frame from (u,v,d)."""
        # Convert to 2D array if it's just a single point
        # We'll remove any singleton dimensions at the end.
        uvz = np.atleast_2d(uvz)

        if not uvz.shape[1] == 3:
            raise ValueError("uvz must have shape (3,) or (N,3)")

        # Now do the actual math
        params = self.cu, self.cv, self.fu, self.fv
        pt_c = _rgbd_triangulate(uvz, params)

        if compute_jacobians:
            jacobians = _rgbd_triangulate_jacobian(uvz, params)

            return np.squeeze(pt_c), np.squeeze(jacobians)

        return np.squeeze(pt_c)

    def __repr__(self):
        return ("{}:\n cu: {:f}\n cv: {:f}\n fu: {:f}\n fv: {:f}\n"
                + "  b: {:f}\n  w: {:d}\n  h: {:d}\n").format(self.__class__.__name__,
                                                              self.cu, self.cv,
                                                              self.fu, self.fv,
                                                              self.w, self.h)


@guvectorize([(float32[:], float32[:], boolean[:]),
              (float64[:], float64[:], boolean[:])],
             '(n),(m)->()', nopython=True, cache=True, target=NUMBA_COMPILATION_TARGET)
def _rgbd_validate(uvz, imshape, out):
    w, h = imshape
    out[0] = (uvz[2] > 0.) & \
        (uvz[1] > 0.) & (uvz[1] < h) & \
        (uvz[0] > 0.) & (uvz[0] < w)


@guvectorize([(float32[:], float32[:], float32[:]),
              (float64[:], float64[:], float64[:])],
             '(n),(m)->(n)', nopython=True, cache=True, target=NUMBA_COMPILATION_TARGET)
def _rgbd_project(pt_c, params, out):
    cu, cv, fu, fv = params

    one_over_z = 1. / pt_c[2]
    out[0] = fu * pt_c[0] * one_over_z + cu
    out[1] = fv * pt_c[1] * one_over_z + cv
    out[2] = pt_c[2]


@guvectorize([(float32[:], float32[:], float32[:, :]),
              (float64[:], float64[:], float64[:, :])],
             '(n),(m)->(n,n)', nopython=True, cache=True, target=NUMBA_COMPILATION_TARGET)
def _rgbd_project_jacobian(pt_c, params, out):
    cu, cv, fu, fv = params

    one_over_z = 1. / pt_c[2]
    one_over_z2 = one_over_z * one_over_z

    # d(u) / d(pt_c)
    out[0, 0] = fu * one_over_z
    out[0, 1] = 0.
    out[0, 2] = -fu * pt_c[0] * one_over_z2

    # d(v) / d(pt_c)
    out[1, 0] = 0.
    out[1, 1] = fv * one_over_z
    out[1, 2] = -fv * pt_c[1] * one_over_z2

    # d(d) / d(pt_c)
    out[2, 0] = 0.
    out[2, 1] = 0.
    out[2, 2] = 1.


@guvectorize([(float32[:], float32[:], float32[:]),
              (float64[:], float64[:], float64[:])],
             '(n),(m)->(n)', nopython=True, cache=True, target=NUMBA_COMPILATION_TARGET)
def _rgbd_triangulate(uvz, params, out):
    cu, cv, fu, fv = params

    out[0] = (uvz[0] - cu) * uvz[2] / fu
    out[1] = (uvz[1] - cv) * uvz[2] / fv
    out[2] = uvz[2]


@guvectorize([(float32[:], float32[:], float32[:, :]),
              (float64[:], float64[:], float64[:, :])],
             '(n),(m)->(n,n)', nopython=True, cache=True, target=NUMBA_COMPILATION_TARGET)
def _rgbd_triangulate_jacobian(uvz, params, out):
    cu, cv, fu, fv = params

    one_over_fu = 1. / fu
    one_over_fv = 1. / fv

    # d(x) / d(uvz)
    out[0, 0] = uvz[2] * one_over_fu
    out[0, 1] = 0.
    out[0, 2] = (uvz[0] - cu) * one_over_fu

    # d(y) / d(uvz)
    out[1, 0] = 0.
    out[1, 1] = uvz[2] * one_over_fv
    out[1, 2] = (uvz[1] - cv) * one_over_fv

    # d(z) / d(uvz)
    out[2, 0] = 0.
    out[2, 1] = 0.
    out[2, 2] = 1.
