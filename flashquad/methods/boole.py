"""Composite Boole's rule."""

import numpy as np

from flashquad.methods.common import _build_grid, _evaluate


def _booles_rule(xp, y, x, axis):
    """Composite Boole's rule along the given axis."""
    n = y.shape[axis]
    if (n - 1) % 4 != 0:
        raise ValueError(
            "Number of points minus one must be a multiple of 4 for Boole's rule."
        )
    dx = (x[-1] - x[0]) / (n - 1)
    c = np.ones(n, dtype=np.float64)
    c[0] = 7
    c[-1] = 7
    c[1:-1:4] = 32
    c[2:-1:4] = 12
    c[3:-1:4] = 32
    c[4:-1:4] = 14
    c = xp.asarray(c, dtype=y.dtype)
    shape = [1] * y.ndim
    shape[axis] = n
    c = xp.reshape(c, tuple(shape))
    return xp.sum(y * c, axis=axis) * 2 * dx / 45


def booles(xp, dtype, func, intervals, num_points, *, params=None, boundary=None):
    """Integrate using composite Boole's rule."""
    ndim = len(intervals)
    grids, expanded_mesh = _build_grid(xp, intervals, num_points, dtype)
    Y = _evaluate(xp, func, expanded_mesh, ndim, params, boundary)

    for dim in reversed(range(ndim)):
        Y = _booles_rule(xp, Y, grids[dim], axis=dim + 1)

    return Y
