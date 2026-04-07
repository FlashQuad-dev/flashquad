import numpy as np
import pytest

from flashquad import (
    trapz_integrate,
    simpson_integrate,
    booles_integrate,
    gauss_integrate,
    mc_integrate,
    adpmc_integrate,
)


# --- Integrands with known analytical results ---

def square(x):
    """x^2, integral on [0,1] = 1/3."""
    return x ** 2


def xy(x, y):
    """x*y, integral on [0,1]^2 = 1/4."""
    return x * y


def parametric_poly(x, params):
    """a*x^2 + b, integral on [0,1] = a/3 + b."""
    a, b = params[0], params[1]
    return a * x ** 2 + b


def unit_disk(x, y):
    """Quarter-unit-disk mask in [0,1]^2."""
    return x ** 2 + y ** 2 < 1.0


PARAMS = np.array([[2.0, 1.0], [3.0, 0.5]])
EXPECTED = PARAMS[:, 0] / 3 + PARAMS[:, 1]


class TestTrapz:
    def test_1d(self):
        result = trapz_integrate(square, [[0, 1]], [1001])
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=1e-4)

    def test_1d_with_params(self):
        result = trapz_integrate(
            parametric_poly, [[0, 1]], [1001], params=PARAMS,
        )
        np.testing.assert_allclose(result, EXPECTED, rtol=1e-4)

    def test_2d(self):
        result = trapz_integrate(xy, [[0, 1], [0, 1]], [101, 101])
        np.testing.assert_allclose(result.item(), 0.25, rtol=1e-3)


class TestSimpson:
    def test_1d(self):
        result = simpson_integrate(square, [[0, 1]], [1001])
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=1e-10)

    def test_1d_with_params(self):
        result = simpson_integrate(
            parametric_poly, [[0, 1]], [1001], params=PARAMS,
        )
        np.testing.assert_allclose(result, EXPECTED, rtol=1e-10)


class TestBooles:
    def test_1d(self):
        result = booles_integrate(square, [[0, 1]], [1001])
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=1e-10)

    def test_1d_with_params(self):
        result = booles_integrate(
            parametric_poly, [[0, 1]], [1001], params=PARAMS,
        )
        np.testing.assert_allclose(result, EXPECTED, rtol=1e-10)


class TestGauss:
    def test_1d(self):
        result = gauss_integrate(square, [[0, 1]], [50])
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=1e-10)

    def test_1d_with_params(self):
        result = gauss_integrate(
            parametric_poly, [[0, 1]], [50], params=PARAMS,
        )
        np.testing.assert_allclose(result, EXPECTED, rtol=1e-10)

    def test_2d(self):
        result = gauss_integrate(xy, [[0, 1], [0, 1]], [20, 20])
        np.testing.assert_allclose(result.item(), 0.25, rtol=1e-10)


class TestMC:
    def test_1d(self):
        np.random.seed(42)
        result = mc_integrate(square, [[0, 1]], 500_000)
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=0.05)

    def test_1d_with_params(self):
        np.random.seed(42)
        result = mc_integrate(
            parametric_poly, [[0, 1]], 500_000, params=PARAMS,
        )
        np.testing.assert_allclose(result, EXPECTED, rtol=0.05)


class TestAdaptiveMC:
    def test_1d(self):
        np.random.seed(42)
        result = adpmc_integrate(
            square, [[0, 1]], 500_000, num_iterations=5,
        )
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=0.05)


class TestBoundary:
    def test_trapz_with_boundary(self):
        result = trapz_integrate(
            xy, [[0, 1], [0, 1]], [201, 201],
            boundary=unit_disk,
        )
        # Integral of x*y over the quarter unit disk = 1/8
        np.testing.assert_allclose(result.item(), 0.125, rtol=0.05)


class TestExplicitBackend:
    def test_xp_parameter(self):
        result = trapz_integrate(
            square, [[0, 1]], [1001], xp=np,
        )
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=1e-4)

    def test_dtype_float32(self):
        result = trapz_integrate(
            square, [[0, 1]], [1001], dtype=np.float32,
        )
        assert result.dtype == np.float32
        np.testing.assert_allclose(result.item(), 1 / 3, rtol=1e-4)
