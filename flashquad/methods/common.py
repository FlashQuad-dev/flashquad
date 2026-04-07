"""Shared helpers for grid-based quadrature methods."""


def _build_grid(xp, intervals, num_points, dtype):
    """Return uniform grids, meshgrid, and batch-expanded mesh."""
    grids = [
        xp.linspace(b[0], b[1], n, dtype=dtype) for b, n in zip(intervals, num_points)
    ]
    mesh = xp.meshgrid(*grids, indexing="ij")
    expanded_mesh = [m[None, ...] for m in mesh]
    return grids, expanded_mesh


def _evaluate(xp, func, expanded_mesh, ndim, params, boundary):
    """Evaluate *func* over the mesh, apply params and boundary."""
    if params is not None:
        vector_length = params.shape[0]
        params_expanded = [
            xp.reshape(params[:, i], (vector_length, *([1] * ndim)))
            for i in range(params.shape[1])
        ]
        Y = func(*expanded_mesh, params_expanded)
    else:
        Y = func(*expanded_mesh)

    if boundary is not None:
        Y = Y * boundary(*expanded_mesh).astype(Y.dtype)

    return Y
