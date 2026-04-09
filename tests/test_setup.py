"""Tests for FlashQuad construction, backend/dtype validation, and repr."""

import subprocess
import sys

import numpy as np
import pytest

from flashquad import FlashQuad
from tests.conftest import gpu, requires_cupy, requires_jax, requires_torch, square


# ---------------------------------------------------------------------------
# Backend validation (always runs — no optional deps needed)
# ---------------------------------------------------------------------------


class TestBackendValidation:
    def test_backend_must_be_string(self):
        with pytest.raises(TypeError, match="backend must be a string"):
            FlashQuad(np)

    def test_unsupported_backend(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            FlashQuad("tensorflow")


# ---------------------------------------------------------------------------
# Dtype validation
# ---------------------------------------------------------------------------


class TestDtypeValidation:
    @requires_torch
    def test_torch_dtype_rejected_for_numpy(self):
        import torch

        with pytest.raises(TypeError, match="not compatible with backend 'numpy'"):
            FlashQuad("numpy", dtype=torch.float32)

    @requires_torch
    def test_numpy_dtype_rejected_for_torch(self):
        with pytest.raises(TypeError, match="not compatible with backend 'torch'"):
            FlashQuad("torch", dtype=np.float32)

    def test_integer_dtype_rejected_for_numpy(self):
        with pytest.raises(TypeError, match="floating-point"):
            FlashQuad("numpy", dtype=np.int32)

    def test_complex_dtype_rejected_for_numpy(self):
        with pytest.raises(TypeError, match="floating-point"):
            FlashQuad("numpy", dtype=np.complex64)


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------


class TestNumpyBackend:
    def test_default_dtype_is_float64(self):
        q = FlashQuad("numpy")
        assert q.dtype == np.float64

    def test_float32_dtype(self):
        q = FlashQuad("numpy", dtype=np.float32)
        result = q.trapz(square, [[0, 1]], [1001])
        assert result.dtype == np.float32
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=1e-4)

    def test_device_is_none(self):
        q = FlashQuad("numpy")
        assert q.device is None


# ---------------------------------------------------------------------------
# Torch backend
# ---------------------------------------------------------------------------


@gpu
@requires_torch
class TestTorchBackend:
    def test_default_dtype_is_float64(self):
        import torch

        tq = FlashQuad("torch")
        assert tq.dtype == torch.float64

    def test_result_is_torch_tensor(self):
        import torch

        tq = FlashQuad("torch")
        result = tq.trapz(square, [[0, 1]], [1001])
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float64
        np.testing.assert_allclose(result.detach().cpu().numpy(), 1 / 3, rtol=1e-4)

    def test_float32_dtype(self):
        import torch

        tq = FlashQuad("torch", dtype=torch.float32)
        result = tq.trapz(square, [[0, 1]], [1001])
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_device_cpu(self):
        import torch

        tq = FlashQuad("torch", device="cpu")
        assert tq.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------


@gpu
@requires_jax
class TestJaxBackend:
    def test_default_dtype_is_float64(self):
        import jax.numpy as jnp

        q = FlashQuad("jax")
        assert q.dtype == jnp.float64

    def test_default_dtype_enables_x64_in_fresh_process(self):
        code = """
import jax
from flashquad import FlashQuad
q = FlashQuad("jax")
r = q.trapz(lambda x: x**2, [[0, 1]], [11])
print(jax.config.jax_enable_x64)
print(r.dtype)
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = proc.stdout.strip().splitlines()
        assert lines == ["True", "float64"]

    def test_result_is_jax_array(self):
        import jax

        q = FlashQuad("jax")
        result = q.trapz(square, [[0, 1]], [1001])
        assert isinstance(result, jax.Array)
        np.testing.assert_allclose(np.asarray(result).item(), 1 / 3, rtol=1e-4)

    def test_float32_dtype(self):
        import jax
        import jax.numpy as jnp

        q = FlashQuad("jax", dtype=jnp.float32)
        result = q.trapz(square, [[0, 1]], [1001])
        assert isinstance(result, jax.Array)
        assert result.dtype == jnp.float32

    def test_mc_respects_explicit_cpu_device(self):
        q = FlashQuad("jax", device="cpu")
        result = q.mc(square, [[0, 1]], 128)
        assert {device.platform for device in result.devices()} == {"cpu"}


# ---------------------------------------------------------------------------
# CuPy backend
# ---------------------------------------------------------------------------


@gpu
@requires_cupy
class TestCupyBackend:
    def test_default_dtype_is_float64(self):
        q = FlashQuad("cupy")
        assert q.dtype == np.float64

    def test_result_is_cupy_array(self):
        import cupy

        q = FlashQuad("cupy")
        result = q.trapz(square, [[0, 1]], [1001])
        assert isinstance(result, cupy.ndarray)
        np.testing.assert_allclose(cupy.asnumpy(result).item(), 1 / 3, rtol=1e-4)

    def test_float32_dtype(self):
        import cupy

        q = FlashQuad("cupy", dtype=np.float32)
        result = q.trapz(square, [[0, 1]], [1001])
        assert isinstance(result, cupy.ndarray)
        assert result.dtype == np.float32

    def test_device_is_none(self):
        q = FlashQuad("cupy")
        assert q.device is None


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_numpy_repr(self):
        q = FlashQuad("numpy")
        r = repr(q)
        assert "numpy" in r
        assert "FlashQuad" in r

    @gpu
    @requires_torch
    def test_torch_repr(self):
        tq = FlashQuad("torch")
        r = repr(tq)
        assert "torch" in r

    @gpu
    @requires_jax
    def test_jax_repr(self):
        q = FlashQuad("jax")
        r = repr(q)
        assert "jax" in r

    @gpu
    @requires_cupy
    def test_cupy_repr(self):
        q = FlashQuad("cupy")
        r = repr(q)
        assert "cupy" in r
