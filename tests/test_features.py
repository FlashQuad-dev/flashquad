"""Tests for flashquad-specific features.

Covers: custom boundary masks, batched parameters, asymmetric
sample-point counts across dimensions, and custom integration intervals.
"""

import numpy as np
import pytest

from tests.conftest import (
    ALL_BACKENDS,
    convert_params,
    parametric_poly,
    square,
    to_numpy,
    unit_disk,
    xy,
)


# ---------------------------------------------------------------------------
# Custom boundary masks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quad", ALL_BACKENDS)
class TestBoundary:
    """Integration over non-rectangular regions via a boundary mask."""

    def test_trapz_quarter_disk(self, quad):
        result = quad.trapz(xy, [[0, 1], [0, 1]], [201, 201], boundary=unit_disk)
        np.testing.assert_allclose(to_numpy(result).item(), 0.125, rtol=0.05)

    def test_gauss_quarter_disk(self, quad):
        result = quad.gauss(xy, [[0, 1], [0, 1]], [50, 50], boundary=unit_disk)
        np.testing.assert_allclose(to_numpy(result).item(), 0.125, rtol=0.05)

    def test_mc_with_boundary(self, quad):
        np.random.seed(42)
        result = quad.mc(xy, [[0, 1], [0, 1]], 500_000, boundary=unit_disk)
        np.testing.assert_allclose(to_numpy(result).item(), 0.125, rtol=0.1)


# ---------------------------------------------------------------------------
# Batched parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quad", ALL_BACKENDS)
class TestBatchedParams:
    """Evaluate the same integrand with multiple parameter sets in one call."""

    def test_trapz_batched(self, quad):
        params = convert_params(quad, parametric_poly.params)
        result = quad.trapz(parametric_poly, [[0, 1]], [1001], params=params)
        np.testing.assert_allclose(
            to_numpy(result), parametric_poly.expected, rtol=1e-4
        )

    def test_simpson_batched(self, quad):
        params = convert_params(quad, parametric_poly.params)
        result = quad.simpson(parametric_poly, [[0, 1]], [1001], params=params)
        np.testing.assert_allclose(
            to_numpy(result), parametric_poly.expected, rtol=1e-5
        )

    def test_gauss_batched(self, quad):
        params = convert_params(quad, parametric_poly.params)
        result = quad.gauss(parametric_poly, [[0, 1]], [50], params=params)
        np.testing.assert_allclose(
            to_numpy(result), parametric_poly.expected, rtol=1e-5
        )

    def test_mc_batched(self, quad):
        np.random.seed(42)
        params = convert_params(quad, parametric_poly.params)
        result = quad.mc(parametric_poly, [[0, 1]], 500_000, params=params)
        np.testing.assert_allclose(
            to_numpy(result), parametric_poly.expected, rtol=0.05
        )

    def test_single_param_row(self, quad):
        single = np.array([[2.0, 1.0]])
        expected = np.array([2.0 / 3 + 1.0])
        params = convert_params(quad, single)
        result = quad.trapz(parametric_poly, [[0, 1]], [1001], params=params)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-4)

    def test_mc_single_param_row_preserves_batch_shape(self, quad):
        np.random.seed(42)
        single = np.array([[2.0, 1.0]])
        expected = np.array([2.0 / 3 + 1.0])
        params = convert_params(quad, single)
        result = quad.mc(parametric_poly, [[0, 1]], 500_000, params=params)
        assert to_numpy(result).shape == (1,)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=0.05)


# ---------------------------------------------------------------------------
# Different sample points per dimension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quad", ALL_BACKENDS)
class TestAsymmetricSamplePoints:
    """Use different numbers of sample points along each axis."""

    def test_trapz_asymmetric_2d(self, quad):
        result = quad.trapz(xy, [[0, 1], [0, 1]], [51, 201])
        np.testing.assert_allclose(to_numpy(result).item(), 0.25, rtol=1e-3)

    def test_gauss_asymmetric_2d(self, quad):
        result = quad.gauss(xy, [[0, 1], [0, 1]], [10, 30])
        np.testing.assert_allclose(to_numpy(result).item(), 0.25, rtol=1e-5)

    def test_trapz_fine_vs_coarse(self, quad):
        """More points in one dim should not degrade the other."""
        coarse = to_numpy(quad.trapz(xy, [[0, 1], [0, 1]], [51, 51])).item()
        fine_y = to_numpy(quad.trapz(xy, [[0, 1], [0, 1]], [51, 501])).item()
        np.testing.assert_allclose(fine_y, 0.25, rtol=1e-4)
        assert abs(fine_y - 0.25) <= abs(coarse - 0.25)


# ---------------------------------------------------------------------------
# Custom integration intervals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quad", ALL_BACKENDS)
class TestCustomIntervals:
    """Verify correct handling of non-unit and non-zero-origin intervals."""

    def test_shifted_interval_1d(self, quad):
        """∫₁² x^2 dx = 7/3."""
        result = quad.trapz(square, [[1, 2]], [10001])
        np.testing.assert_allclose(to_numpy(result).item(), 7 / 3, rtol=1e-5)

    def test_wide_interval_gauss(self, quad):
        """∫₀¹⁰ x^2 dx = 1000/3."""
        result = quad.gauss(square, [[0, 10]], [50])
        np.testing.assert_allclose(to_numpy(result).item(), 1000 / 3, rtol=1e-10)

    def test_2d_non_unit_intervals(self, quad):
        """∫₀²∫₀³ x*y dy dx = 9."""
        result = quad.gauss(xy, [[0, 2], [0, 3]], [20, 20])
        np.testing.assert_allclose(to_numpy(result).item(), 9.0, rtol=1e-5)
