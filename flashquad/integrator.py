"""FlashQuad — numerical integrator with pluggable array backends."""

from flashquad.utils import _resolve_backend_namespace, _default_dtype, _validate_dtype
from flashquad.methods import trapz as _trapz
from flashquad.methods import simpson as _simpson
from flashquad.methods import booles as _booles
from flashquad.methods import gauss as _gauss
from flashquad.methods import mc as _mc
from flashquad.methods import adpmc as _adpmc


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
        return _trapz(
            self.xp, self.dtype, func, intervals, num_points,
            params=params, boundary=boundary,
        )

    def simpson(
        self, func, intervals, num_points, *,
        params=None, boundary=None,
    ):
        """Integrate using composite Simpson's 1/3 rule.

        Args: See :meth:`trapz`. Each entry in *num_points* must be odd.
        """
        return _simpson(
            self.xp, self.dtype, func, intervals, num_points,
            params=params, boundary=boundary,
        )

    def booles(
        self, func, intervals, num_points, *,
        params=None, boundary=None,
    ):
        """Integrate using composite Boole's rule.

        Args: See :meth:`trapz`. ``(num_points[i] - 1)`` must be divisible by 4.
        """
        return _booles(
            self.xp, self.dtype, func, intervals, num_points,
            params=params, boundary=boundary,
        )

    def gauss(
        self, func, intervals, num_points, *,
        params=None, boundary=None,
    ):
        """Integrate using Gauss-Legendre quadrature.

        Args: See :meth:`trapz`.
        """
        return _gauss(
            self.xp, self.dtype, func, intervals, num_points,
            params=params, boundary=boundary,
        )

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
        return _mc(
            self.xp, self.dtype, func, intervals, num_points,
            params=params, boundary=boundary,
        )

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
        return _adpmc(
            self.xp, self.dtype, func, intervals, num_points,
            params=params, boundary=boundary, num_iterations=num_iterations,
        )
