"""Backend-native random number generation."""

import numpy as np

from flashquad.utils import _to_device


def _rand(xp, shape, dtype, device):
    """Generate uniform random samples on the target device when possible."""
    try:
        import torch

        if isinstance(device, torch.device):
            return torch.rand(shape, dtype=dtype, device=device)
    except ImportError:
        pass
    try:
        import cupy

        if hasattr(xp, "__name__") and "cupy" in xp.__name__:
            return cupy.random.random(shape, dtype=dtype)
    except ImportError:
        pass
    try:
        import jax
        import jax.random

        if hasattr(xp, "__name__") and "jax" in xp.__name__:
            key = jax.random.PRNGKey(np.random.randint(0, 2**31))
            return _to_device(jax.random.uniform(key, shape, dtype=dtype), device)
    except ImportError:
        pass
    return _to_device(xp.asarray(np.random.rand(*shape), dtype=dtype), device)
