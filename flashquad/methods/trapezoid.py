"""Composite trapezoidal rule."""

import numpy as np

from flashquad.methods.common import _build_grid, _evaluate


def _trapezoid(xp, y, x, axis):
    """Composite trapezoidal rule along the given axis."""
    n = y.shape[axis]
    dx = (x[-1] - x[0]) / (n - 1)
    c = np.ones(n, dtype=np.float64)
    c[0] = 0.5
    c[-1] = 0.5
    c = xp.asarray(c, dtype=y.dtype)
    shape = [1] * y.ndim
    shape[axis] = n
    c = xp.reshape(c, tuple(shape))
    return xp.sum(y * c, axis=axis) * dx


def trapz(xp, dtype, func, intervals, num_points, *, params=None, boundary=None):
    """Integrate using the composite trapezoidal rule."""
    ndim = len(intervals)
    grids, expanded_mesh = _build_grid(xp, intervals, num_points, dtype)
    Y = _evaluate(xp, func, expanded_mesh, ndim, params, boundary)

    for dim in reversed(range(ndim)):
        Y = _trapezoid(xp, Y, grids[dim], axis=dim + 1)

    return Y
