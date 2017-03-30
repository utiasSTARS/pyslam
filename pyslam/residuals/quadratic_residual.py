import numpy as np


class QuadraticResidual:
    """A simple quadratic residual for fitting a parabola to data."""

    def __init__(self, x, y, stiffness):
        self.x = np.array([x])
        self.y = np.array([y])
        self.stiffness = np.array([stiffness])

    def evaluate(self, params, compute_jacobians=None):
        residual = self.stiffness * (params[0] * self.x * self.x
                                     + params[1] * self.x
                                     + params[2]
                                     - self.y)

        if compute_jacobians:
            jacobians = [None for _ in enumerate(params)]

            if compute_jacobians[0]:
                jacobians[0] = self.stiffness * self.x * self.x

            if compute_jacobians[1]:
                jacobians[1] = self.stiffness * self.x

            if compute_jacobians[2]:
                jacobians[2] = self.stiffness * 1.

            return residual, np.squeeze(jacobians)

        return residual
