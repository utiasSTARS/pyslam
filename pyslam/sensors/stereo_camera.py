import numpy as np
from numba import guvectorize, float32, float64, boolean

NUMBA_COMPILATION_TARGET = 'parallel'


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

    def clone(self):
        return self.__class__(self.cu, self.cv,
                              self.fu, self.fv, self.b,
                              self.w, self.h)

    def compute_pixel_grid(self):
        self.u_grid, self.v_grid = np.meshgrid(range(0, self.w),
                                               range(0, self.h), indexing='xy')
        self.u_grid = self.u_grid.astype(float)
        self.v_grid = self.v_grid.astype(float)

    def is_valid_measurement(self, uvd):
        """Check if one or more uvd measurements is valid.
           Returns indices of valid measurements.
        """
        uvd = np.atleast_2d(uvd)

        if not uvd.shape[1] == 3:
            raise ValueError("uvd must have shape (3,) or (N,3)")

        return _stereo_validate(uvd, [self.w, self.h])

    def project(self, pt_c, compute_jacobians=None):
        """Project 3D point(s) in the sensor frame into (u,v,d) coordinates."""
        # Convert to 2D array if it's just a single point.
        # We'll remove any singleton dimensions at the end.
        pt_c = np.atleast_2d(pt_c)

        if not pt_c.shape[1] == 3:
            raise ValueError("pt_c must have shape (3,) or (N,3)")

        # Now do the actual math
        params = self.cu, self.cv, self.fu, self.fv, self.b
        uvd = _stereo_project(pt_c, params)

        if compute_jacobians:
            jacobians = _stereo_project_jacobian(pt_c, params)

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
        params = self.cu, self.cv, self.fu, self.fv, self.b
        pt_c = _stereo_triangulate(uvd, params)

        if compute_jacobians:
            jacobians = _stereo_triangulate_jacobian(uvd, params)

            return np.squeeze(pt_c), np.squeeze(jacobians)

        return np.squeeze(pt_c)

    def __repr__(self):
        return ("{}:\n cu: {:f}\n cv: {:f}\n fu: {:f}\n fv: {:f}\n"
                + "  b: {:f}\n  w: {:d}\n  h: {:d}\n").format(self.__class__.__name__,
                                                              self.cu, self.cv,
                                                              self.fu, self.fv,
                                                              self.b,
                                                              self.w, self.h)


@guvectorize([(float32[:], float32[:], boolean[:]),
              (float64[:], float64[:], boolean[:])],
             '(n),(m)->()', nopython=True, cache=True, target=NUMBA_COMPILATION_TARGET)
def _stereo_validate(uvd, imshape, out):
    w, h = imshape
    out[0] = (uvd[2] > 0.) & (uvd[2] < w) & \
        (uvd[1] > 0.) & (uvd[1] < h) & \
        (uvd[0] > 0.) & (uvd[0] < w)


@guvectorize([(float32[:], float32[:], float32[:]),
              (float64[:], float64[:], float64[:])],
             '(n),(m)->(n)', nopython=True, cache=True, target=NUMBA_COMPILATION_TARGET)
def _stereo_project(pt_c, params, out):
    cu, cv, fu, fv, b = params

    one_over_z = 1. / pt_c[2]
    out[0] = fu * pt_c[0] * one_over_z + cu
    out[1] = fv * pt_c[1] * one_over_z + cv
    out[2] = fu * b * one_over_z


@guvectorize([(float32[:], float32[:], float32[:, :]),
              (float64[:], float64[:], float64[:, :])],
             '(n),(m)->(n,n)', nopython=True, cache=True, target=NUMBA_COMPILATION_TARGET)
def _stereo_project_jacobian(pt_c, params, out):
    cu, cv, fu, fv, b = params

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
    out[2, 2] = -fu * b * one_over_z2


@guvectorize([(float32[:], float32[:], float32[:]),
              (float64[:], float64[:], float64[:])],
             '(n),(m)->(n)', nopython=True, cache=True, target=NUMBA_COMPILATION_TARGET)
def _stereo_triangulate(uvd, params, out):
    cu, cv, fu, fv, b = params

    b_over_d = b / uvd[2]
    fu_over_fv = fu / fv

    out[0] = (uvd[0] - cu) * b_over_d
    out[1] = (uvd[1] - cv) * b_over_d * fu_over_fv
    out[2] = fu * b_over_d


@guvectorize([(float32[:], float32[:], float32[:, :]),
              (float64[:], float64[:], float64[:, :])],
             '(n),(m)->(n,n)', nopython=True, cache=True, target=NUMBA_COMPILATION_TARGET)
def _stereo_triangulate_jacobian(uvd, params, out):
    cu, cv, fu, fv, b = params

    b_over_d = b / uvd[2]
    b_over_d2 = b_over_d / uvd[2]
    fu_over_fv = fu / fv

    # d(x) / d(uvd)
    out[0, 0] = b_over_d
    out[0, 1] = 0.
    out[0, 2] = (cu - uvd[0]) * b_over_d2

    # d(y) / d(uvd)
    out[1, 0] = 0.
    out[1, 1] = b_over_d * fu_over_fv
    out[1, 2] = (cv - uvd[1]) * b_over_d2 * fu_over_fv

    # d(z) / d(uvd)
    out[2, 0] = 0.
    out[2, 1] = 0.
    out[2, 2] = -fu * b_over_d2
