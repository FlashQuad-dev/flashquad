"""Shared fixtures and test integrands for flashquad tests."""

import numpy as np
import pytest

from flashquad import FlashQuad

# Enable JAX float64 support before any JAX computation.
try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# pytest CLI: --no-gpu
# ---------------------------------------------------------------------------

_GPU_BACKENDS = frozenset({"torch", "cupy", "jax"})


def pytest_addoption(parser):
    parser.addoption(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Skip all tests that require a GPU (torch, cupy, jax).",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--no-gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="--no-gpu flag set")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Backend availability detection
# ---------------------------------------------------------------------------


def _try_backend(name, **kwargs):
    """Try to construct a FlashQuad instance for *name*.

    Returns the instance on success, or None when the backend's
    dependencies are missing.
    """
    try:
        return FlashQuad(name, **kwargs)
    except (ImportError, ModuleNotFoundError):
        return None


_BACKEND_DEFS = [
    ("numpy", {}),
    ("torch", {}),
    ("jax", {}),
    ("cupy", {}),
]

ALL_BACKENDS = []
for _name, _kw in _BACKEND_DEFS:
    _inst = _try_backend(_name, **_kw)
    _marks = [pytest.mark.gpu] if _name in _GPU_BACKENDS else []
    if _inst is not None:
        ALL_BACKENDS.append(pytest.param(_inst, id=_name, marks=_marks))
    else:
        _marks.append(pytest.mark.skip(reason=f"{_name} not installed"))
        ALL_BACKENDS.append(pytest.param(_name, id=_name, marks=_marks))


def backend_available(name):
    """Return True if *name* can be constructed as a flashquad backend."""
    return _try_backend(name) is not None


requires_torch = pytest.mark.skipif(
    not backend_available("torch"), reason="torch not installed"
)
requires_jax = pytest.mark.skipif(
    not backend_available("jax"), reason="jax not installed"
)
requires_cupy = pytest.mark.skipif(
    not backend_available("cupy"), reason="cupy not installed"
)
gpu = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Integrands with known analytical results
# ---------------------------------------------------------------------------


def square(x):
    """x^2, integral on [0,1] = 1/3."""
    return x**2


def xy(x, y):
    """x*y, integral on [0,1]^2 = 1/4."""
    return x * y


def sin_1d(x):
    """sin(x), integral on [0, pi] = 2."""
    xp = _infer_namespace(x)
    return xp.sin(x)


def exp_1d(x):
    """e^x, integral on [0,1] = e - 1."""
    xp = _infer_namespace(x)
    return xp.exp(x)


def parametric_poly(x, a, b):
    """a*x^2 + b, integral on [0,1] = a/3 + b."""
    return a * x**2 + b


parametric_poly.params = np.array([[2.0, 1.0], [3.0, 0.5]])
parametric_poly.expected = (
    parametric_poly.params[:, 0] / 3 + parametric_poly.params[:, 1]
)


def gaussian_1d(x):
    """e^(-x^2), integral on [-3,3] = sqrt(pi)*erf(3)."""
    xp = _infer_namespace(x)
    return xp.exp(-(x**2))


def rational_1d(x):
    """1/(1+x^2), integral on [0,1] = pi/4."""
    return 1.0 / (1.0 + x**2)


def sincos_2d(x, y):
    """sin(x)*cos(y), integral on [0,pi]x[0,pi/2] = 2."""
    xp = _infer_namespace(x)
    return xp.sin(x) * xp.cos(y)


def sum_of_squares_2d(x, y):
    """x^2 + y^2, integral on [0,1]^2 = 2/3."""
    return x**2 + y**2


def exp_neg_r2(x, y):
    """e^(-(x^2+y^2)), integral on [0,2]^2 = (pi/4)*erf(2)^2."""
    xp = _infer_namespace(x)
    return xp.exp(-(x**2 + y**2))


def xyz_3d(x, y, z):
    """x*y*z, integral on [0,1]^3 = 1/8."""
    return x * y * z


def unit_disk(x, y):
    """Quarter-unit-disk mask in [0,1]^2."""
    return x**2 + y**2 < 1.0


def _infer_namespace(x):
    """Return the array namespace for *x*."""
    try:
        import array_api_compat

        return array_api_compat.array_namespace(x)
    except Exception:
        return np


# ---------------------------------------------------------------------------
# Cross-backend conversion helpers
# ---------------------------------------------------------------------------


def to_numpy(x):
    """Convert any backend tensor to a numpy array."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    try:
        import cupy

        if isinstance(x, cupy.ndarray):
            return cupy.asnumpy(x)
    except ImportError:
        pass
    return np.asarray(x)


def convert_params(quad, params):
    """Convert numpy *params* to the tensor type expected by *quad*'s backend."""
    try:
        import torch

        if isinstance(quad.dtype, torch.dtype):
            return torch.tensor(params, dtype=quad.dtype, device=quad.device)
    except ImportError:
        pass

    if quad._backend_name == "cupy":
        import cupy

        return cupy.asarray(params, dtype=quad.dtype)

    if quad._backend_name == "jax":
        import jax.numpy as jnp

        return jnp.asarray(params, dtype=quad.dtype)

    return np.asarray(params, dtype=quad.dtype)
