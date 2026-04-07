import pytest
import numpy as np
import torch

from flashquad import FlashQuad


# --- Integrands with known analytical results ---


def square(x):
    """x^2, integral on [0,1] = 1/3."""
    return x**2


def xy(x, y):
    """x*y, integral on [0,1]^2 = 1/4."""
    return x * y


def parametric_poly(x, params):
    """a*x^2 + b, integral on [0,1] = a/3 + b."""
    a, b = params[0], params[1]
    return a * x**2 + b


def unit_disk(x, y):
    """Quarter-unit-disk mask in [0,1]^2."""
    return x**2 + y**2 < 1.0


PARAMS = np.array([[2.0, 1.0], [3.0, 0.5]])
EXPECTED = PARAMS[:, 0] / 3 + PARAMS[:, 1]

quad = FlashQuad("numpy")


class TestTrapz:
    def test_1d(self):
        result = quad.trapz(square, [[0, 1]], [1001])
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=1e-4)

    def test_1d_with_params(self):
        result = quad.trapz(
            parametric_poly,
            [[0, 1]],
            [1001],
            params=PARAMS,
        )
        np.testing.assert_allclose(result, EXPECTED, rtol=1e-4)

    def test_2d(self):
        result = quad.trapz(xy, [[0, 1], [0, 1]], [101, 101])
        np.testing.assert_allclose(result.item(), 0.25, rtol=1e-3)


class TestSimpson:
    def test_1d(self):
        result = quad.simpson(square, [[0, 1]], [1001])
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=1e-5)

    def test_1d_with_params(self):
        result = quad.simpson(
            parametric_poly,
            [[0, 1]],
            [1001],
            params=PARAMS,
        )
        np.testing.assert_allclose(result, EXPECTED, rtol=1e-5)


class TestBooles:
    def test_1d(self):
        result = quad.booles(square, [[0, 1]], [1001])
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=1e-5)

    def test_1d_with_params(self):
        result = quad.booles(
            parametric_poly,
            [[0, 1]],
            [1001],
            params=PARAMS,
        )
        np.testing.assert_allclose(result, EXPECTED, rtol=1e-5)


class TestGauss:
    def test_1d(self):
        result = quad.gauss(square, [[0, 1]], [50])
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=1e-5)

    def test_1d_with_params(self):
        result = quad.gauss(
            parametric_poly,
            [[0, 1]],
            [50],
            params=PARAMS,
        )
        np.testing.assert_allclose(result, EXPECTED, rtol=1e-5)

    def test_2d(self):
        result = quad.gauss(xy, [[0, 1], [0, 1]], [20, 20])
        np.testing.assert_allclose(result.item(), 0.25, rtol=1e-5)


class TestMC:
    def test_1d(self):
        np.random.seed(42)
        result = quad.mc(square, [[0, 1]], 500_000)
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=0.05)

    def test_1d_with_params(self):
        np.random.seed(42)
        result = quad.mc(
            parametric_poly,
            [[0, 1]],
            500_000,
            params=PARAMS,
        )
        np.testing.assert_allclose(result, EXPECTED, rtol=0.05)


class TestAdaptiveMC:
    def test_1d(self):
        np.random.seed(42)
        result = quad.adpmc(
            square,
            [[0, 1]],
            500_000,
            num_iterations=5,
        )
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=0.05)


class TestBoundary:
    def test_trapz_with_boundary(self):
        result = quad.trapz(
            xy,
            [[0, 1], [0, 1]],
            [201, 201],
            boundary=unit_disk,
        )
        np.testing.assert_allclose(result.item(), 0.125, rtol=0.05)


class TestTorchBackend:
    def test_torch_backend(self):
        tq = FlashQuad("torch")
        result = tq.trapz(square, [[0, 1]], [1001])
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float64
        np.testing.assert_allclose(
            result.detach().cpu().numpy(),
            1 / 3,
            rtol=1e-4,
        )

    def test_torch_float32(self):
        tq = FlashQuad("torch", dtype=torch.float32)
        result = tq.trapz(square, [[0, 1]], [1001])
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32


class TestNumpyDtype:
    def test_numpy_float32(self):
        q = FlashQuad("numpy", dtype=np.float32)
        result = q.trapz(square, [[0, 1]], [1001])
        assert result.dtype == np.float32
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=1e-4)


class TestDtypeValidation:
    def test_torch_dtype_rejected_for_numpy(self):
        with pytest.raises(TypeError, match="not compatible with backend 'numpy'"):
            FlashQuad("numpy", dtype=torch.float32)

    def test_numpy_dtype_rejected_for_torch(self):
        with pytest.raises(TypeError, match="not compatible with backend 'torch'"):
            FlashQuad("torch", dtype=np.float32)

    def test_backend_must_be_string(self):
        with pytest.raises(TypeError, match="backend must be a string"):
            FlashQuad(np)

    def test_unsupported_backend(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            FlashQuad("tensorflow")


class TestRepr:
    def test_repr(self):
        q = FlashQuad("numpy")
        assert "numpy" in repr(q)
