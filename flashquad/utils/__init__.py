"""General utility functions."""

from flashquad.utils.backend import (
    _BACKEND_MODULES,
    _TORCH_BACKENDS,
    _NUMPY_COMPAT_BACKENDS,
    _resolve_backend_namespace,
    _default_dtype,
    _validate_dtype,
)

__all__ = [
    "_BACKEND_MODULES",
    "_TORCH_BACKENDS",
    "_NUMPY_COMPAT_BACKENDS",
    "_resolve_backend_namespace",
    "_default_dtype",
    "_validate_dtype",
]
