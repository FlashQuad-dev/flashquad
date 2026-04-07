"""Numerical integration with pluggable array backends via Python Array API."""

import importlib

import numpy as np


_BACKEND_MODULES = {
    "numpy": "array_api_compat.numpy",
    "torch": "array_api_compat.torch",
    "cupy": "array_api_compat.cupy",
    "jax": "jax.numpy",
}

_TORCH_BACKENDS = frozenset({"torch"})
_NUMPY_COMPAT_BACKENDS = frozenset({"numpy", "cupy", "jax"})


def _resolve_backend_namespace(backend):
    """Resolve a backend string to its array namespace."""
    try:
        return importlib.import_module(_BACKEND_MODULES[backend])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backend {backend!r}. Expected one of "
            f"{', '.join(sorted(_BACKEND_MODULES))}."
        ) from exc


def _default_dtype(xp):
    """Return the namespace-native default floating dtype."""
    return getattr(xp, "float64", np.float64)


def _validate_dtype(backend_name, dtype):
    """Raise TypeError when *dtype* is incompatible with *backend_name*."""
    if backend_name in _TORCH_BACKENDS:
        import torch

        if not isinstance(dtype, torch.dtype):
            raise TypeError(
                f"dtype {dtype!r} is not compatible with backend 'torch'. "
                f"Use a torch dtype (e.g., torch.float32, torch.float64)."
            )
        return

    try:
        import torch

        if isinstance(dtype, torch.dtype):
            raise TypeError(
                f"dtype {dtype!r} is not compatible with backend {backend_name!r}. "
                f"Use a numpy-compatible dtype (e.g., numpy.float32, numpy.float64)."
            )
    except ImportError:
        pass

    try:
        np.dtype(dtype)
    except TypeError:
        raise TypeError(
            f"dtype {dtype!r} is not compatible with backend {backend_name!r}."
        )


# ---------------------------------------------------------------------------
# Low-level quadrature kernels
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Common helpers for building grids and evaluating integrands
# ---------------------------------------------------------------------------

def _build_grid(xp, intervals, num_points, dtype):
    """Return uniform grids, meshgrid, and batch-expanded mesh."""
    grids = [
        xp.linspace(b[0], b[1], n, dtype=dtype)
        for b, n in zip(intervals, num_points)
    ]
    mesh = xp.meshgrid(*grids, indexing='ij')
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


# ---------------------------------------------------------------------------
# FlashQuad — public integrator class
# ---------------------------------------------------------------------------

class FlashQuad:
    """Numerical integrator bound to a specific array backend and dtype.

    Parameters
    ----------
    backend : str
        Array backend name. One of ``'numpy'``, ``'torch'``, ``'cupy'``,
        ``'jax'``.
    dtype : optional
        Floating-point dtype that must be compatible with the chosen backend.
        For ``'torch'`` pass a ``torch.dtype`` (e.g. ``torch.float64``);
        for numpy-compatible backends pass a numpy dtype (e.g. ``numpy.float64``).
        Defaults to the backend's ``float64``.
    """

    def __init__(self, backend="numpy", dtype=None):
        if not isinstance(backend, str):
            raise TypeError(
                f"backend must be a string, got {type(backend).__name__}"
            )
        self._backend_name = backend
        self.xp = _resolve_backend_namespace(backend)
        if dtype is not None:
            _validate_dtype(backend, dtype)
            self.dtype = dtype
        else:
            self.dtype = _default_dtype(self.xp)

    def __repr__(self):
        return (
            f"FlashQuad(backend={self._backend_name!r}, dtype={self.dtype!r})"
        )

    # ---- integration methods ------------------------------------------------

    def trapz(
        self, func, intervals, num_points, *,
        params=None, boundary=None,
    ):
        """Integrate using the composite trapezoidal rule.

        Args:
            func: Integrand. Called as ``func(*mesh_coords)`` or
                ``func(*mesh_coords, params_expanded)`` when params is given.
            intervals: Integration bounds per dimension, e.g. ``[[0, 1], [0, 1]]``.
            num_points: Grid points per dimension, e.g. ``[101, 101]``.
            params: Parameter array shaped ``(batch, num_params)``. Each row is
                one set of parameters for batched evaluation.
            boundary: Optional mask function applied to the integrand.
        """
        xp, dtype = self.xp, self.dtype
        ndim = len(intervals)
        grids, expanded_mesh = _build_grid(xp, intervals, num_points, dtype)
        Y = _evaluate(xp, func, expanded_mesh, ndim, params, boundary)

        for dim in reversed(range(ndim)):
            Y = _trapezoid(xp, Y, grids[dim], axis=dim + 1)

        return Y

    def simpson(
        self, func, intervals, num_points, *,
        params=None, boundary=None,
    ):
        """Integrate using composite Simpson's 1/3 rule.

        Args: See :meth:`trapz`. Each entry in *num_points* must be odd.
        """
        xp, dtype = self.xp, self.dtype
        ndim = len(intervals)
        grids, expanded_mesh = _build_grid(xp, intervals, num_points, dtype)
        Y = _evaluate(xp, func, expanded_mesh, ndim, params, boundary)

        for dim in reversed(range(ndim)):
            Y = _simpsons_rule(xp, Y, grids[dim], axis=dim + 1)

        return Y

    def booles(
        self, func, intervals, num_points, *,
        params=None, boundary=None,
    ):
        """Integrate using composite Boole's rule.

        Args: See :meth:`trapz`. ``(num_points[i] - 1)`` must be divisible by 4.
        """
        xp, dtype = self.xp, self.dtype
        ndim = len(intervals)
        grids, expanded_mesh = _build_grid(xp, intervals, num_points, dtype)
        Y = _evaluate(xp, func, expanded_mesh, ndim, params, boundary)

        for dim in reversed(range(ndim)):
            Y = _booles_rule(xp, Y, grids[dim], axis=dim + 1)

        return Y

    def gauss(
        self, func, intervals, num_points, *,
        params=None, boundary=None,
    ):
        """Integrate using Gauss-Legendre quadrature.

        Args: See :meth:`trapz`.
        """
        xp, dtype = self.xp, self.dtype
        ndim = len(intervals)

        grids, weights = [], []
        for bound, n in zip(intervals, num_points):
            nodes, w = _gauss_nodes_weights(xp, n, dtype)
            a, b = bound[0], bound[1]
            grids.append(0.5 * (b - a) * nodes + 0.5 * (b + a))
            weights.append(0.5 * (b - a) * w)

        mesh = xp.meshgrid(*grids, indexing='ij')
        expanded_mesh = [m[None, ...] for m in mesh]
        Y = _evaluate(xp, func, expanded_mesh, ndim, params, boundary)

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

    def mc(
        self, func, intervals, num_points, *,
        params=None, boundary=None,
    ):
        """Integrate using Monte Carlo sampling.

        Args:
            func: Integrand function.
            intervals: Integration bounds per dimension.
            num_points: Total number of random samples (single integer).
            params: Parameter array shaped ``(batch, num_params)``.
            boundary: Optional mask function.
        """
        xp, dtype = self.xp, self.dtype
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

    def adpmc(
        self, func, intervals, num_points, *,
        params=None, boundary=None, num_iterations=10,
    ):
        """Integrate using iterative Monte Carlo sampling.

        Args:
            func: Integrand function.
            intervals: Integration bounds per dimension.
            num_points: Random samples per iteration (single integer).
            params: Parameter array shaped ``(batch, num_params)``.
            boundary: Optional mask function.
            num_iterations: Number of sampling iterations.
        """
        xp, dtype = self.xp, self.dtype
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
