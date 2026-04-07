"""Accuracy tests comparing flashquad against scipy.integrate.

We integrate the same functions with both flashquad and scipy and verify
that the results agree within tight tolerance.
"""

import math

import numpy as np
import pytest
from scipy import integrate, special

from flashquad import FlashQuad
from tests.conftest import (
    exp_1d,
    exp_neg_r2,
    gaussian_1d,
    rational_1d,
    sin_1d,
    sincos_2d,
    square,
    sum_of_squares_2d,
    xy,
    xyz_3d,
)

quad = FlashQuad("numpy")


# ---------------------------------------------------------------------------
# 1-D comparisons
# ---------------------------------------------------------------------------

_FUNCTIONS_1D = [
    pytest.param(square, [0, 1], id="x^2_on_0_1"),
    pytest.param(sin_1d, [0, np.pi], id="sin_on_0_pi"),
    pytest.param(exp_1d, [0, 1], id="exp_on_0_1"),
    pytest.param(gaussian_1d, [-3, 3], id="gaussian_on_-3_3"),
    pytest.param(rational_1d, [0, 1], id="rational_on_0_1"),
]

_METHODS_1D = [
    pytest.param("trapz", [10001], 1e-6, id="trapz"),
    pytest.param("simpson", [10001], 1e-8, id="simpson"),
    pytest.param("booles", [10001], 1e-10, id="booles"),
    pytest.param("gauss", [100], 1e-12, id="gauss"),
]


@pytest.mark.parametrize("func, interval", _FUNCTIONS_1D)
@pytest.mark.parametrize("method_name, num_points, rtol", _METHODS_1D)
def test_1d_vs_scipy(func, interval, method_name, num_points, rtol):
    def scalar_func(x):
        return float(func(np.asarray(x)))

    scipy_val, _ = integrate.quad(scalar_func, *interval)

    method = getattr(quad, method_name)
    fq_val = method(func, [interval], num_points).item()

    np.testing.assert_allclose(fq_val, scipy_val, rtol=rtol)


# ---------------------------------------------------------------------------
# 2-D comparisons — simple
# ---------------------------------------------------------------------------


def _scipy_dblquad(func_yx, x_lo, x_hi, y_lo, y_hi):
    """Wrapper around dblquad (note: dblquad takes f(y,x))."""
    val, _ = integrate.dblquad(func_yx, x_lo, x_hi, y_lo, y_hi)
    return val


_FUNCTIONS_2D = [
    pytest.param(
        xy,
        lambda y, x: x * y,
        [[0, 1], [0, 1]],
        id="xy_on_unit",
    ),
    pytest.param(
        sincos_2d,
        lambda y, x: math.sin(x) * math.cos(y),
        [[0, np.pi], [0, np.pi / 2]],
        id="sincos_on_rect",
    ),
    pytest.param(
        sum_of_squares_2d,
        lambda y, x: x**2 + y**2,
        [[0, 1], [0, 1]],
        id="x2_plus_y2_on_unit",
    ),
    pytest.param(
        exp_neg_r2,
        lambda y, x: math.exp(-(x**2 + y**2)),
        [[0, 2], [0, 2]],
        id="exp_neg_r2_on_0_2",
    ),
]

_METHODS_2D = [
    pytest.param("trapz", [201, 201], 1e-4, id="trapz"),
    pytest.param("gauss", [30, 30], 1e-10, id="gauss"),
]


@pytest.mark.parametrize("fq_func, scipy_func, intervals", _FUNCTIONS_2D)
@pytest.mark.parametrize("method_name, num_points, rtol", _METHODS_2D)
def test_2d_vs_scipy(fq_func, scipy_func, intervals, method_name, num_points, rtol):
    scipy_val = _scipy_dblquad(scipy_func, *intervals[0], *intervals[1])

    method = getattr(quad, method_name)
    fq_val = method(fq_func, intervals, num_points).item()

    np.testing.assert_allclose(fq_val, scipy_val, rtol=rtol)


# ---------------------------------------------------------------------------
# 3-D comparisons
# ---------------------------------------------------------------------------


def test_3d_gauss_vs_scipy_tplquad():
    """∫∫∫ x*y*z over [0,1]^3 = 1/8."""
    scipy_val, _ = integrate.tplquad(
        lambda z, y, x: x * y * z,
        0,
        1,
        0,
        1,
        0,
        1,
    )
    fq_val = quad.gauss(xyz_3d, [[0, 1], [0, 1], [0, 1]], [10, 10, 10]).item()
    np.testing.assert_allclose(fq_val, scipy_val, rtol=1e-10)


def test_3d_trapz_vs_scipy_tplquad():
    """∫∫∫ x*y*z over [0,1]^3 = 1/8."""
    scipy_val, _ = integrate.tplquad(
        lambda z, y, x: x * y * z,
        0,
        1,
        0,
        1,
        0,
        1,
    )
    fq_val = quad.trapz(xyz_3d, [[0, 1], [0, 1], [0, 1]], [51, 51, 51]).item()
    np.testing.assert_allclose(fq_val, scipy_val, rtol=1e-3)


def test_3d_exp_gauss_vs_scipy():
    """∫∫∫ e^(-(x^2+y^2+z^2)) over [0,1]^3."""

    def fq_func(x, y, z):
        xp = _infer_ns(x)
        return xp.exp(-(x**2 + y**2 + z**2))

    scipy_val, _ = integrate.tplquad(
        lambda z, y, x: math.exp(-(x**2 + y**2 + z**2)),
        0,
        1,
        0,
        1,
        0,
        1,
    )
    fq_val = quad.gauss(fq_func, [[0, 1], [0, 1], [0, 1]], [15, 15, 15]).item()
    np.testing.assert_allclose(fq_val, scipy_val, rtol=1e-10)


# ---------------------------------------------------------------------------
# 1-D: known analytical values as sanity cross-check with scipy
# ---------------------------------------------------------------------------


class TestAnalyticalCrossCheck:
    """Verify our results and scipy agree with known closed-form values."""

    def test_gaussian_integral(self):
        """∫_{-3}^{3} e^{-x^2} dx = sqrt(pi) * erf(3)."""
        analytical = math.sqrt(math.pi) * special.erf(3)
        fq_val = quad.gauss(gaussian_1d, [[-3, 3]], [80]).item()
        scipy_val, _ = integrate.quad(lambda x: math.exp(-(x**2)), -3, 3)

        np.testing.assert_allclose(fq_val, analytical, rtol=1e-14)
        np.testing.assert_allclose(scipy_val, analytical, rtol=1e-14)

    def test_arctan_integral(self):
        """∫_0^1 1/(1+x^2) dx = pi/4."""
        analytical = math.pi / 4
        fq_val = quad.gauss(rational_1d, [[0, 1]], [50]).item()
        scipy_val, _ = integrate.quad(lambda x: 1 / (1 + x**2), 0, 1)

        np.testing.assert_allclose(fq_val, analytical, rtol=1e-14)
        np.testing.assert_allclose(scipy_val, analytical, rtol=1e-14)

    def test_2d_sincos(self):
        """∫_0^pi ∫_0^{pi/2} sin(x)*cos(y) dy dx = 2."""
        fq_val = quad.gauss(
            sincos_2d,
            [[0, np.pi], [0, np.pi / 2]],
            [30, 30],
        ).item()
        np.testing.assert_allclose(fq_val, 2.0, rtol=1e-10)

    def test_3d_xyz(self):
        """∫_0^1 ∫_0^1 ∫_0^1 x*y*z dz dy dx = 1/8."""
        fq_val = quad.gauss(xyz_3d, [[0, 1], [0, 1], [0, 1]], [10, 10, 10]).item()
        np.testing.assert_allclose(fq_val, 0.125, rtol=1e-10)


# ---------------------------------------------------------------------------
# Monte Carlo vs scipy (loose tolerance)
# ---------------------------------------------------------------------------


def test_mc_vs_scipy():
    np.random.seed(42)

    scipy_val, _ = integrate.quad(lambda x: x**2, 0, 1)
    fq_val = quad.mc(square, [[0, 1]], 1_000_000).item()

    np.testing.assert_allclose(fq_val, scipy_val, rtol=0.02)


def test_mc_2d_vs_scipy():
    np.random.seed(42)

    scipy_val, _ = integrate.dblquad(lambda y, x: x * y, 0, 1, 0, 1)
    fq_val = quad.mc(xy, [[0, 1], [0, 1]], 1_000_000).item()

    np.testing.assert_allclose(fq_val, scipy_val, rtol=0.02)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_ns(x):
    try:
        import array_api_compat

        return array_api_compat.array_namespace(x)
    except Exception:
        return np
