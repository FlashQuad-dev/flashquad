"""Shared helpers for grid-based quadrature methods."""

from flashquad.utils import _to_device


def _build_grid(xp, intervals, num_points, dtype, device=None):
    """Return uniform grids, meshgrid, and batch-expanded mesh."""
    grids = [
        _to_device(xp.linspace(b[0], b[1], n, dtype=dtype), device)
        for b, n in zip(intervals, num_points)
    ]
    mesh = xp.meshgrid(*grids, indexing="ij")
    expanded_mesh = [m[None, ...] for m in mesh]
    return grids, expanded_mesh


def _evaluate(xp, func, expanded_mesh, ndim, params, boundary, device=None):
    """Evaluate *func* over the mesh, apply params and boundary."""
    if params is not None:
        params = _to_device(params, device)
        vector_length = params.shape[0]
        params_expanded = [
            xp.reshape(params[:, i], (vector_length, *([1] * ndim)))
            for i in range(params.shape[1])
        ]
        Y = func(*expanded_mesh, *params_expanded)
    else:
        Y = func(*expanded_mesh)

    if boundary is not None:
        Y = Y * boundary(*expanded_mesh)

    return Y
