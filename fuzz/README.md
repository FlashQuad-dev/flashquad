# fuzz

Random-integrand fuzz tests comparing flashquad against scipy and torchquad.
Separate from `tests/` — not collected by pytest.

## Install

```bash
uv sync --extra fuzz
```

Pulls in `scipy`, `torch`, and `torchquad`.

## Run

```bash
uv run --extra fuzz python -m fuzz.run_fuzz --dims 1-10 --trials 10
```

Useful flags:

| flag | default | meaning |
|------|---------|---------|
| `--dims` | `1-10` | single dim (`3`) or inclusive range (`1-10`) |
| `--trials` | `5` | trials per dim |
| `--seed` | `0` | RNG seed (integrand, intervals, MC) |
| `--scipy-max-dim` | `3` | largest dim using `scipy.nquad` as reference |
| `--rtol-gauss` | `1e-4` | relative tolerance for Gauss–Legendre (dim ≤ 5) |
| `--rtol-mc` | `5e-2` | relative tolerance for Monte Carlo (dim ≥ 6) |
| `--atol-gauss` | `1e-8` | absolute floor for Gauss (near-zero guard) |
| `--atol-mc` | `5e-3` | absolute floor for MC (near-zero guard) |

Pairs pass when `|a-b| <= atol + rtol * max(|a|, |b|)` (numpy-style). When the
scipy reference is present it is treated as ground truth — only the other
backends are checked against scipy, never against each other. Exits non-zero if
any pair fails.

## Design

- Integrands are random sums of polynomial, sin/cos, and Gaussian-bump terms.
  The same callable dispatches on numpy arrays (flashquad), torch tensors
  (torchquad), and Python scalars (scipy) via `array_api_compat`.
- Low dim (≤ 5): Gauss–Legendre on both flashquad and torchquad with matched
  points-per-dim; scipy's iterated quadrature as ground truth when dim ≤ 3.
- High dim (≥ 6): Monte Carlo on both sides with the same seed, compared
  pairwise — no cheap exact reference at this scale.

## Files

- `integrands.py` — `RandomIntegrand`, `random_intervals`
- `runners.py` — backend adapters
- `run_fuzz.py` — CLI
