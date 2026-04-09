"""Backend resolution and dtype validation."""

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


def _enable_jax_x64():
    """Enable JAX x64 mode when available and currently disabled."""
    import jax

    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)


def _default_dtype(xp, backend_name=None):
    """Return the namespace-native default floating dtype."""
    if backend_name == "jax":
        _enable_jax_x64()
    return getattr(xp, "float64")


def _validate_dtype(backend_name, dtype):
    """Raise TypeError when *dtype* is incompatible with *backend_name*."""
    if backend_name in _TORCH_BACKENDS:
        import torch

        if not isinstance(dtype, torch.dtype):
            raise TypeError(
                f"dtype {dtype!r} is not compatible with backend 'torch'. "
                f"Use a torch dtype (e.g., torch.float32, torch.float64)."
            )
        if not dtype.is_floating_point:
            raise TypeError(
                f"dtype {dtype!r} is not compatible with backend 'torch'. "
                "Use a floating-point torch dtype."
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
        resolved_dtype = np.dtype(dtype)
    except TypeError:
        raise TypeError(
            f"dtype {dtype!r} is not compatible with backend {backend_name!r}."
        )
    if not np.issubdtype(resolved_dtype, np.floating):
        raise TypeError(
            f"dtype {dtype!r} is not compatible with backend {backend_name!r}. "
            "Use a floating-point numpy-compatible dtype."
        )
    if backend_name == "jax" and resolved_dtype == np.dtype(np.float64):
        _enable_jax_x64()


def _resolve_device(backend_name, device):
    """Resolve compute device for the given backend.

    For ``'torch'`` and ``'jax'``, auto-selects GPU when available if
    *device* is ``None``.  Other backends return ``None`` (cupy is
    inherently GPU; numpy is CPU-only).
    """
    if backend_name in _TORCH_BACKENDS:
        import torch

        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device) if isinstance(device, str) else device

    if backend_name == "jax":
        import jax

        if device is None:
            try:
                gpus = jax.devices("gpu")
                return gpus[0] if gpus else jax.devices("cpu")[0]
            except RuntimeError:
                return jax.devices("cpu")[0]
        if isinstance(device, str):
            return jax.devices(device)[0]
        return device

    return None


def _to_device(tensor, device):
    """Move *tensor* to *device* when the backend supports it."""
    if device is None:
        return tensor
    if hasattr(tensor, "to"):
        return tensor.to(device=device)
    try:
        import jax

        if isinstance(tensor, jax.Array):
            return jax.device_put(tensor, device)
    except ImportError:
        pass
    return tensor
