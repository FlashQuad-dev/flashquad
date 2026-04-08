"""Accuracy tests against known analytical results.

For each integrand with a closed-form solution we test every quadrature
method and every backend that is available in the test environment.
"""

import numpy as np
import pytest

from tests.conftest import (
    ALL_BACKENDS,
    convert_params,
    exp_1d,
    parametric_poly,
    sin_1d,
    square,
    to_numpy,
    xy,
)

# ---------------------------------------------------------------------------
# Deterministic methods — 1-D
# ---------------------------------------------------------------------------

_GRID_1D = {
    "trapz": {"num_points": [1001], "rtol": 1e-4},
    "simpson": {"num_points": [1001], "rtol": 1e-5},
    "booles": {"num_points": [1001], "rtol": 1e-5},
    "gauss": {"num_points": [50], "rtol": 1e-10},
}


@pytest.mark.parametrize("quad", ALL_BACKENDS)
@pytest.mark.parametrize("method_name", list(_GRID_1D))
class TestAnalytical1D:
    """x^2 on [0,1] = 1/3 for every deterministic method."""

    def test_square(self, quad, method_name):
        cfg = _GRID_1D[method_name]
        method = getattr(quad, method_name)
        result = method(square, [[0, 1]], cfg["num_points"])
        np.testing.assert_allclose(to_numpy(result).item(), 1 / 3, rtol=cfg["rtol"])

    def test_sin(self, quad, method_name):
        cfg = _GRID_1D[method_name]
        method = getattr(quad, method_name)
        result = method(sin_1d, [[0, np.pi]], cfg["num_points"])
        np.testing.assert_allclose(to_numpy(result).item(), 2.0, rtol=cfg["rtol"])

    def test_exp(self, quad, method_name):
        cfg = _GRID_1D[method_name]
        method = getattr(quad, method_name)
        result = method(exp_1d, [[0, 1]], cfg["num_points"])
        np.testing.assert_allclose(to_numpy(result).item(), np.e - 1, rtol=cfg["rtol"])


# ---------------------------------------------------------------------------
# Deterministic methods — 2-D
# ---------------------------------------------------------------------------

_GRID_2D = {
    "trapz": {"num_points": [101, 101], "rtol": 1e-3},
    "gauss": {"num_points": [20, 20], "rtol": 1e-5},
}


@pytest.mark.parametrize("quad", ALL_BACKENDS)
@pytest.mark.parametrize("method_name", list(_GRID_2D))
class TestAnalytical2D:
    """x*y on [0,1]^2 = 1/4 for methods that support 2-D."""

    def test_xy(self, quad, method_name):
        cfg = _GRID_2D[method_name]
        method = getattr(quad, method_name)
        result = method(xy, [[0, 1], [0, 1]], cfg["num_points"])
        np.testing.assert_allclose(to_numpy(result).item(), 0.25, rtol=cfg["rtol"])


# ---------------------------------------------------------------------------
# Monte Carlo methods — 1-D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quad", ALL_BACKENDS)
class TestAnalyticalMC:
    """Monte Carlo methods with generous tolerance."""

    def test_mc_square(self, quad):
        np.random.seed(42)
        result = quad.mc(square, [[0, 1]], 500_000)
        np.testing.assert_allclose(to_numpy(result).item(), 1 / 3, rtol=0.05)


# ---------------------------------------------------------------------------
# Batched parameter tests — all deterministic methods
# ---------------------------------------------------------------------------

_PARAM_GRID = {
    "trapz": {"num_points": [1001], "rtol": 1e-4},
    "simpson": {"num_points": [1001], "rtol": 1e-5},
    "booles": {"num_points": [1001], "rtol": 1e-5},
    "gauss": {"num_points": [50], "rtol": 1e-5},
}


@pytest.mark.parametrize("quad", ALL_BACKENDS)
@pytest.mark.parametrize("method_name", list(_PARAM_GRID))
class TestParametricAccuracy:
    """a*x^2 + b on [0,1] = a/3 + b with batched parameters."""

    def test_parametric_poly(self, quad, method_name):
        cfg = _PARAM_GRID[method_name]
        method = getattr(quad, method_name)
        params = convert_params(quad, parametric_poly.params)
        result = method(parametric_poly, [[0, 1]], cfg["num_points"], params=params)
        np.testing.assert_allclose(
            to_numpy(result), parametric_poly.expected, rtol=cfg["rtol"]
        )


@pytest.mark.parametrize("quad", ALL_BACKENDS)
class TestParametricAccuracyMC:
    """Batched parameters for MC methods."""

    def test_mc_parametric(self, quad):
        np.random.seed(42)
        params = convert_params(quad, parametric_poly.params)
        result = quad.mc(parametric_poly, [[0, 1]], 500_000, params=params)
        np.testing.assert_allclose(
            to_numpy(result), parametric_poly.expected, rtol=0.05
        )
