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
    from torchquad import GaussLegendre

    dim = len(intervals)
    total_n = points_per_dim**dim
    domain = torch.tensor(intervals, dtype=torch.float64)
    wrapped = _torch_wrap(fn)
    gl = GaussLegendre()
    val = gl.integrate(wrapped, dim=dim, N=total_n, integration_domain=domain)
    return float(val.item() if hasattr(val, "item") else val)


def run_torchquad_mc(fn, intervals, n: int, seed: int) -> float:
    import torch

    _setup_torchquad()
    from torchquad import MonteCarlo

    dim = len(intervals)
    domain = torch.tensor(intervals, dtype=torch.float64)
    wrapped = _torch_wrap(fn)
    mc = MonteCarlo()
    val = mc.integrate(wrapped, dim=dim, N=n, integration_domain=domain, seed=seed)
    return float(val.item() if hasattr(val, "item") else val)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


_TORCHQUAD_READY = False


def _setup_torchquad():
    """Configure torchquad once, silencing its chatty loguru output."""
    global _TORCHQUAD_READY
    if _TORCHQUAD_READY:
        return
    os.environ.setdefault("LOGURU_LEVEL", "ERROR")
    from loguru import logger as _logger
    _logger.remove()
    from torchquad import set_up_backend
    set_up_backend("torch", data_type="float64")
    _TORCHQUAD_READY = True


def _torch_wrap(fn):
    """Adapt a flashquad-style ``fn(*cols)`` integrand to torchquad's ``fn(x)``.

    torchquad passes a single ``(N, dim)`` tensor; flashquad-style integrands
    expect one array per dimension.
    """
    def wrapped(x):
        cols = [x[..., i] for i in range(x.shape[-1])]
        return fn(*cols)

    return wrapped


@contextlib.contextmanager
def _suppress_stderr():
    """Silence IntegrationWarning spam from scipy.nquad during fuzzing."""
    import warnings

    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore", _scipy_integrate.IntegrationWarning)
        yield
