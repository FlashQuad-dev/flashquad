"""Numerical integration with pluggable array backends via Python Array API."""

import numpy as np
import array_api_compat


def _resolve_backend(params, xp, dtype):
    """Determine the array namespace and dtype from inputs."""
    if xp is None:
        xp = (
            array_api_compat.array_namespace(params)
            if params is not None
            else np
        )
    if dtype is None:
        dtype = params.dtype if params is not None else xp.float64
    return xp, dtype


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


def _booles_rule(xp, y, x, axis):
    """Composite Boole's rule along the given axis."""
    n = y.shape[axis]
    if (n - 1) % 4 != 0:
        raise ValueError(
            "Number of points minus one must be a multiple of 4 "
            "for Boole's rule."
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


def _gauss_nodes_weights(xp, n, dtype):
    """Compute Gauss-Legendre nodes and weights via numpy, then convert."""
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n)
    return xp.asarray(nodes_np, dtype=dtype), xp.asarray(weights_np, dtype=dtype)


# Method 1: Trapezoidal rule
def trapz_integrate(
    func, intervals, num_points, *,
    params=None, boundary=None, xp=None, dtype=None,
):
    """
    Integrate using the composite trapezoidal rule.

    Args:
        func: Integrand. Called as ``func(*mesh_coords)`` or
            ``func(*mesh_coords, params_expanded)`` when params is given.
        intervals: Integration bounds per dimension, e.g. ``[[0, 1], [0, 1]]``.
        num_points: Grid points per dimension, e.g. ``[101, 101]``.
        params: Parameter array shaped ``(batch, num_params)``. Each row is
            one set of parameters for batched evaluation.
        boundary: Optional mask function applied to the integrand.
        xp: Array namespace (inferred from *params* when omitted).
        dtype: Floating-point dtype (inferred from *params* when omitted).
    """
    xp, dtype = _resolve_backend(params, xp, dtype)
    ndim = len(intervals)

    grids = [
        xp.linspace(b[0], b[1], n, dtype=dtype)
        for b, n in zip(intervals, num_points)
    ]
    mesh = xp.meshgrid(*grids, indexing='ij')
    expanded_mesh = [m[None, ...] for m in mesh]

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

    for dim in reversed(range(ndim)):
        Y = _trapezoid(xp, Y, grids[dim], axis=dim + 1)

    return Y


# Method 2: Simpson's rule
def simpson_integrate(
    func, intervals, num_points, *,
    params=None, boundary=None, xp=None, dtype=None,
):
    """
    Integrate using composite Simpson's 1/3 rule.

    Args: See ``trapz_integrate``. Each entry in *num_points* must be odd.
    """
    xp, dtype = _resolve_backend(params, xp, dtype)
    ndim = len(intervals)

    grids = [
        xp.linspace(b[0], b[1], n, dtype=dtype)
        for b, n in zip(intervals, num_points)
    ]
    mesh = xp.meshgrid(*grids, indexing='ij')
    expanded_mesh = [m[None, ...] for m in mesh]

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

    for dim in reversed(range(ndim)):
        Y = _simpsons_rule(xp, Y, grids[dim], axis=dim + 1)

    return Y


# Method 3: Boole's rule
def booles_integrate(
    func, intervals, num_points, *,
    params=None, boundary=None, xp=None, dtype=None,
):
    """
    Integrate using composite Boole's rule.

    Args: See ``trapz_integrate``. ``(num_points[i] - 1)`` must be
        divisible by 4.
    """
    xp, dtype = _resolve_backend(params, xp, dtype)
    ndim = len(intervals)

    grids = [
        xp.linspace(b[0], b[1], n, dtype=dtype)
        for b, n in zip(intervals, num_points)
    ]
    mesh = xp.meshgrid(*grids, indexing='ij')
    expanded_mesh = [m[None, ...] for m in mesh]

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

    for dim in reversed(range(ndim)):
        Y = _booles_rule(xp, Y, grids[dim], axis=dim + 1)

    return Y


# Method 4: Gauss-Legendre quadrature
def gauss_integrate(
    func, intervals, num_points, *,
    params=None, boundary=None, xp=None, dtype=None,
):
    """
    Integrate using Gauss-Legendre quadrature.

    Args: See ``trapz_integrate``.
    """
    xp, dtype = _resolve_backend(params, xp, dtype)
    ndim = len(intervals)

    grids, weights = [], []
    for bound, n in zip(intervals, num_points):
        nodes, w = _gauss_nodes_weights(xp, n, dtype)
        a, b = bound[0], bound[1]
        grids.append(0.5 * (b - a) * nodes + 0.5 * (b + a))
        weights.append(0.5 * (b - a) * w)

    mesh = xp.meshgrid(*grids, indexing='ij')
    expanded_mesh = [m[None, ...] for m in mesh]

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

    for dim in reversed(range(ndim)):
        w = weights[dim]
        shape = (
            [1] * (dim + 1)
            + [w.shape[0]]
            + [1] * (Y.ndim - dim - 2)
        )
        w = xp.reshape(w, tuple(shape))
        Y = xp.sum(Y * w, axis=dim + 1)

    return Y


# Method 5: Monte Carlo integration
def mc_integrate(
    func, intervals, num_points, *,
    params=None, boundary=None, xp=None, dtype=None,
):
    """
    Integrate using Monte Carlo sampling.

    Args:
        func: Integrand function.
        intervals: Integration bounds per dimension.
        num_points: Total number of random samples (single integer).
        params: Parameter array shaped ``(batch, num_params)``.
        boundary: Optional mask function.
        xp: Array namespace.
        dtype: Floating-point dtype.
    """
    xp, dtype = _resolve_backend(params, xp, dtype)
    vector_length = params.shape[0] if params is not None else 1

    samples = []
    volume = 1.0
    for bound in intervals:
        a, b = bound[0], bound[1]
        raw = np.random.rand(vector_length, num_points)
        sample = xp.asarray(raw, dtype=dtype) * (b - a) + a
        samples.append(sample[..., None])
        volume *= b - a

    if params is not None:
        params_expanded = [
            xp.reshape(params[:, i], (vector_length, 1, 1))
            for i in range(params.shape[1])
        ]
        Y = func(*samples, params_expanded)
    else:
        Y = func(*samples)

    if boundary is not None:
        Y = Y * boundary(*samples).astype(Y.dtype)

    return volume * xp.mean(Y, axis=1).squeeze()


# Method 6: Adaptive Monte Carlo integration
def adpmc_integrate(
    func, intervals, num_points, *,
    params=None, boundary=None, num_iterations=10,
    xp=None, dtype=None,
):
    """
    Integrate using iterative Monte Carlo sampling.

    Args:
        func: Integrand function.
        intervals: Integration bounds per dimension.
        num_points: Random samples per iteration (single integer).
        params: Parameter array shaped ``(batch, num_params)``.
        boundary: Optional mask function.
        num_iterations: Number of sampling iterations.
        xp: Array namespace.
        dtype: Floating-point dtype.
    """
    xp, dtype = _resolve_backend(params, xp, dtype)
    vector_length = params.shape[0] if params is not None else 1

    volume = 1.0
    for bound in intervals:
        volume *= bound[1] - bound[0]

    for _ in range(num_iterations):
        samples = []
        for bound in intervals:
            a, b = bound[0], bound[1]
            raw = np.random.rand(vector_length, num_points)
            sample = xp.asarray(raw, dtype=dtype) * (b - a) + a
            samples.append(sample[..., None])

        if params is not None:
            params_expanded = [
                xp.reshape(params[:, i], (vector_length, 1, 1))
                for i in range(params.shape[1])
            ]
            Y = func(*samples, params_expanded)
        else:
            Y = func(*samples)

        if boundary is not None:
            Y = Y * boundary(*samples).astype(Y.dtype)

    return volume * xp.mean(Y, axis=1).squeeze()
