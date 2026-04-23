"""Backend adapters used by the fuzz runner.

Each ``run_*`` function takes the same ``(fn, intervals)`` inputs and returns a
Python float. Methods are chosen to be roughly comparable across backends:
Gauss–Legendre for low-dim (deterministic), Monte Carlo for high-dim.
"""

import contextlib
import io
import os

import numpy as np
from scipy import integrate as _scipy_integrate

from flashquad import FlashQuad


_fq = FlashQuad("numpy")


def run_flashquad_gauss(fn, intervals, points_per_dim: int = 12) -> float:
    dim = len(intervals)
    return float(_fq.gauss(fn, intervals, [points_per_dim] * dim).item())


def run_flashquad_mc(fn, intervals, n: int, seed: int) -> float:
    # flashquad's mc uses the numpy global RNG — seed it for reproducibility.
    np.random.seed(seed)
    return float(_fq.mc(fn, intervals, n).item())


def run_scipy(fn, intervals, opts=None) -> float:
    """scipy.integrate.nquad. Only practical up to dim ~3-4."""
    opts = opts or {}
    with _suppress_stderr():
        val, _err = _scipy_integrate.nquad(fn, intervals, opts=opts)
    return float(val)


def run_torchquad_gauss(fn, intervals, points_per_dim: int = 12) -> float:
    import torch

    _setup_torchquad()

    dim = len(intervals)
    total_n = points_per_dim**dim
    domain = torch.tensor(intervals, dtype=torch.float64)
    wrapped = _torch_wrap(fn)
    gl = _fixed_gauss_legendre()
    val = gl.integrate(wrapped, dim=dim, N=total_n, integration_domain=domain)
    return _torchquad_scalar(val)


def run_torchquad_mc(fn, intervals, n: int, seed: int) -> float:
    import torch

    _setup_torchquad()
    from torchquad import MonteCarlo

    dim = len(intervals)
    domain = torch.tensor(intervals, dtype=torch.float64)
    wrapped = _torch_wrap(fn)
    mc = MonteCarlo()
    val = mc.integrate(wrapped, dim=dim, N=n, integration_domain=domain, seed=seed)
    return _torchquad_scalar(val)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


_TORCHQUAD_READY = False


def _setup_torchquad():
    """Configure torchquad once, disabling its chatty loguru output."""
    global _TORCHQUAD_READY
    if _TORCHQUAD_READY:
        return
    os.environ.setdefault("LOGURU_LEVEL", "ERROR")
    import torch
    from loguru import logger as _logger
    _logger.remove()
    from torchquad import set_up_backend
    set_up_backend(
        "torch",
        data_type="float64",
        torch_enable_cuda=torch.cuda.is_available(),
    )
    _TORCHQUAD_READY = True


def _fixed_gauss_legendre():
    """Return GaussLegendre with torch weight construction that avoids copy warnings."""
    import torch
    from torchquad import GaussLegendre

    class _FixedGaussLegendre(GaussLegendre):
        def _weights(self, N, dim, backend, requires_grad=False):
            if backend != "torch":
                return super()._weights(N, dim, backend, requires_grad)

            device = (
                torch.get_default_device()
                if hasattr(torch, "get_default_device")
                else None
            )
            weights = torch.as_tensor(
                self._cached_points_and_weights(N)[1],
                dtype=torch.get_default_dtype(),
                device=device,
            )
            if requires_grad:
                weights.requires_grad_(True)
            mesh = torch.meshgrid(*([weights] * dim), indexing="ij")
            return torch.prod(torch.stack(mesh, dim=0), dim=0).ravel()

    return _FixedGaussLegendre()


def _torch_wrap(fn):
    """Adapt a flashquad-style ``fn(*cols)`` integrand to torchquad's ``fn(x)``.

    torchquad passes a single ``(N, dim)`` tensor; flashquad-style integrands
    expect one array per dimension.
    """
    def wrapped(x):
        cols = [x[..., i] for i in range(x.shape[-1])]
        return _torchquad_vector(fn(*cols), x)

    return wrapped


def _torchquad_vector(values, x):
    """Return a non-deprecated torchquad integrand shape for scalar values."""
    if not hasattr(values, "shape"):
        values = x.new_full((x.shape[0],), float(values))
    elif len(values.shape) == 0:
        values = values.expand(x.shape[0])

    if len(values.shape) == 1:
        return values.unsqueeze(-1).repeat(1, 2)
    if len(values.shape) == 2 and values.shape[1] == 1:
        return values.repeat(1, 2)
    return values


def _torchquad_scalar(val) -> float:
    if hasattr(val, "reshape"):
        val = val.reshape(-1)[0]
    return float(val.item() if hasattr(val, "item") else val)


@contextlib.contextmanager
def _suppress_stderr():
    """Silence IntegrationWarning spam from scipy.nquad during fuzzing."""
    import warnings

    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore", _scipy_integrate.IntegrationWarning)
        yield
