"""Composite Simpson's 1/3 rule."""

import numpy as np

from flashquad.methods.common import _build_grid, _evaluate


def _simpsons_rule(xp, y, x, axis):
    """Composite Simpson's 1/3 rule along the given axis."""
    n = y.shape[axis]
    if n % 2 == 0:
        raise ValueError(
            "Number of points must be odd for Simpson's rule."
        )
    dx = (x[-1] - x[0]) / (n - 1)
    c = np.ones(n, dtype=np.float64)
    c[1:-1:2] = 4
    c[2:-1:2] = 2
    c = xp.asarray(c, dtype=y.dtype)
    shape = [1] * y.ndim
    shape[axis] = n
    c = xp.reshape(c, tuple(shape))
    return xp.sum(y * c, axis=axis) * dx / 3


def simpson(xp, dtype, func, intervals, num_points, *, params=None, boundary=None):
    """Integrate using composite Simpson's 1/3 rule."""
    ndim = len(intervals)
    grids, expanded_mesh = _build_grid(xp, intervals, num_points, dtype)
    Y = _evaluate(xp, func, expanded_mesh, ndim, params, boundary)

    for dim in reversed(range(ndim)):
        Y = _simpsons_rule(xp, Y, grids[dim], axis=dim + 1)

    return Y
