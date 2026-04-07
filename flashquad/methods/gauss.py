"""Gauss-Legendre quadrature."""

import numpy as np

from flashquad.methods.common import _evaluate


def _gauss_nodes_weights(xp, n, dtype):
    """Compute Gauss-Legendre nodes and weights via numpy, then convert."""
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n)
    return xp.asarray(nodes_np, dtype=dtype), xp.asarray(weights_np, dtype=dtype)


def gauss(xp, dtype, func, intervals, num_points, *, params=None, boundary=None):
    """Integrate using Gauss-Legendre quadrature."""
    ndim = len(intervals)

    grids, weights = [], []
    for bound, n in zip(intervals, num_points):
        nodes, w = _gauss_nodes_weights(xp, n, dtype)
        a, b = bound[0], bound[1]
        grids.append(0.5 * (b - a) * nodes + 0.5 * (b + a))
        weights.append(0.5 * (b - a) * w)

    mesh = xp.meshgrid(*grids, indexing="ij")
    expanded_mesh = [m[None, ...] for m in mesh]
    Y = _evaluate(xp, func, expanded_mesh, ndim, params, boundary)

    for dim in reversed(range(ndim)):
        w = weights[dim]
        shape = [1] * (dim + 1) + [w.shape[0]] + [1] * (Y.ndim - dim - 2)
        w = xp.reshape(w, tuple(shape))
        Y = xp.sum(Y * w, axis=dim + 1)

    return Y
