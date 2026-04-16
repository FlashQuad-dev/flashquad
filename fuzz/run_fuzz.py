"""Fuzz runner: compare flashquad, scipy and torchquad on random integrands.

Run directly::

    uv run --extra fuzz python -m fuzz.run_fuzz --dims 1-10 --trials 10

Integrands are smooth random sums of polynomial, trig, and Gaussian bumps. For
dimensions with a tractable ground truth (``<= --scipy-max-dim``) scipy's
iterated quadrature is used as reference; beyond that, flashquad and torchquad
are compared pairwise only.
"""

import argparse
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np

from fuzz.integrands import RandomIntegrand, random_intervals
from fuzz.runners import (
    run_flashquad_gauss,
    run_flashquad_mc,
    run_scipy,
    run_torchquad_gauss,
    run_torchquad_mc,
)


# Gauss becomes expensive as points_per_dim**dim; switch to MC past this dim.
_GAUSS_MAX_DIM = 5
_GAUSS_POINTS_PER_DIM = 12
_MC_SAMPLES = 500_000


@dataclass
class TrialResult:
    dim: int
    trial: int
    intervals: list
    signature: str
    values: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def _parse_dims(arg: str) -> list[int]:
    if "-" in arg:
        lo, hi = arg.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(arg)]


def _pair_diag(a: float, b: float, rtol: float, atol: float) -> tuple[bool, str]:
    """Test ``|a-b| <= atol + rtol*max(|a|,|b|)`` (numpy-style) and format a reason."""
    diff = abs(a - b)
    tol = atol + rtol * max(abs(a), abs(b))
    denom = max(abs(a), abs(b), 1e-300)
    rel = diff / denom
    ok = diff <= tol
    msg = f"|diff|={diff:.3e} (rel={rel:.3e}), tol={tol:.3e}"
    return ok, msg


def _eval_all(fn, intervals, dim: int, mc_seed: int) -> dict:
    """Compute each available backend's estimate of the integral."""
    values: dict = {}
    if dim <= _GAUSS_MAX_DIM:
        values["flashquad"] = run_flashquad_gauss(
            fn, intervals, _GAUSS_POINTS_PER_DIM
        )
        values["torchquad"] = run_torchquad_gauss(
            fn, intervals, _GAUSS_POINTS_PER_DIM
        )
    else:
        values["flashquad"] = run_flashquad_mc(fn, intervals, _MC_SAMPLES, mc_seed)
        values["torchquad"] = run_torchquad_mc(fn, intervals, _MC_SAMPLES, mc_seed)
    return values


def _compare(values: dict, rtol: float, atol: float) -> list[str]:
    """Flag pairs that violate ``|a-b| <= atol + rtol*max(|a|,|b|)``.

    When a ``scipy`` value is present it is treated as ground truth: only the
    scipy–vs–other pairs are checked, never backend-vs-backend. Otherwise all
    pairs are checked.
    """
    warns: list[str] = []
    if "scipy" in values:
        ref = values["scipy"]
        for name, val in values.items():
            if name == "scipy":
                continue
            ok, msg = _pair_diag(val, ref, rtol, atol)
            if not ok:
                warns.append(f"{name} vs scipy: {msg}")
        return warns
    keys = list(values)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            ok, msg = _pair_diag(values[ka], values[kb], rtol, atol)
            if not ok:
                warns.append(f"{ka} vs {kb}: {msg}")
    return warns


def run_trial(
    dim: int,
    trial_idx: int,
    rng: np.random.Generator,
    scipy_max_dim: int,
    rtol_gauss: float,
    rtol_mc: float,
    atol_gauss: float,
    atol_mc: float,
) -> TrialResult:
    fn = RandomIntegrand(dim, rng)
    intervals = random_intervals(dim, rng)
    mc_seed = int(rng.integers(0, 2**31 - 1))

    values = _eval_all(fn, intervals, dim, mc_seed)
    if dim <= scipy_max_dim:
        values["scipy"] = run_scipy(fn, intervals)

    if dim <= _GAUSS_MAX_DIM:
        rtol, atol = rtol_gauss, atol_gauss
    else:
        rtol, atol = rtol_mc, atol_mc
    warns = _compare(values, rtol, atol)

    return TrialResult(
        dim=dim,
        trial=trial_idx,
        intervals=intervals,
        signature=fn.describe(),
        values=values,
        warnings=warns,
    )


def _format_intervals(intervals):
    return [(round(a, 2), round(b, 2)) for a, b in intervals]


def _print_trial(result: TrialResult) -> None:
    print(f"\n[dim={result.dim}, trial={result.trial}]")
    print(f"  intervals: {_format_intervals(result.intervals)}")
    print(f"  integrand: {result.signature}")
    for name, val in result.values.items():
        print(f"  {name:<10} = {val:+.6e}")
    for w in result.warnings:
        print(f"  WARN: {w}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dims", default="1-10", help="Single dim or range, e.g. '3' or '1-10'."
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scipy-max-dim",
        type=int,
        default=3,
        help="Largest dim for which scipy.nquad is used as ground truth.",
    )
    parser.add_argument(
        "--rtol-gauss",
        type=float,
        default=1e-4,
        help="Relative tolerance for deterministic (Gauss) comparisons.",
    )
    parser.add_argument(
        "--rtol-mc",
        type=float,
        default=5e-2,
        help="Relative tolerance for Monte Carlo comparisons.",
    )
    parser.add_argument(
        "--atol-gauss",
        type=float,
        default=1e-8,
        help="Absolute tolerance floor for Gauss comparisons (near-zero guard).",
    )
    parser.add_argument(
        "--atol-mc",
        type=float,
        default=5e-3,
        help="Absolute tolerance floor for MC comparisons (near-zero guard).",
    )
    args = parser.parse_args(argv)

    dims = _parse_dims(args.dims)
    rng = np.random.default_rng(args.seed)

    total_warn = 0
    for d in dims:
        header = f" dim={d} "
        print("\n" + header.center(60, "="))
        for t in range(args.trials):
            result = run_trial(
                d,
                t,
                rng,
                args.scipy_max_dim,
                args.rtol_gauss,
                args.rtol_mc,
                args.atol_gauss,
                args.atol_mc,
            )
            _print_trial(result)
            for w in result.warnings:
                warnings.warn(f"[dim={d}, trial={t}] {w}", stacklevel=2)
            total_warn += len(result.warnings)

    print(f"\nTotal warnings: {total_warn}")
    return 0 if total_warn == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
