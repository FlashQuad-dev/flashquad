# API reference

## {py:obj}`flashquad.integrator.FlashQuad`

The main entry point. Create an instance bound to a backend, then call any
integration method.

```python
from flashquad import FlashQuad

fq = FlashQuad(backend="numpy")
fq = FlashQuad(backend="torch", dtype=torch.float32, device="cuda:0")
```

**Parameters:**

- **backend** (`str`) — One of `"numpy"`, `"torch"`, `"cupy"`, `"jax"`.
- **dtype** (optional) — Floating-point dtype compatible with the backend. Defaults to `float64`.
- **device** (optional) — Compute device (relevant for `"torch"` and `"jax"`). Auto-selects GPU when available.

### Integration methods

All methods share a common signature:

```python
fq.method(func, intervals, num_points, *, params=None, boundary=None)
```

| Method | Description | Constraints |
|--------|-------------|-------------|
| {py:obj}`~flashquad.integrator.FlashQuad.trapz` | Composite trapezoidal rule | — |
| {py:obj}`~flashquad.integrator.FlashQuad.simpson` | Composite Simpson's 1/3 rule | `num_points` must be odd |
| {py:obj}`~flashquad.integrator.FlashQuad.booles` | Composite Boole's rule | `(num_points - 1)` divisible by 4 |
| {py:obj}`~flashquad.integrator.FlashQuad.gauss` | Gauss-Legendre quadrature | — |
| {py:obj}`~flashquad.integrator.FlashQuad.mc` | Monte Carlo sampling | `num_points` is a single int |

**Common arguments:**

- **func** — Integrand callable. Called as `func(*coords)` or `func(*coords, params)`.
- **intervals** — Integration bounds per dimension, e.g. `[[0, 1], [0, 1]]`.
- **num_points** — Grid points per dimension (list), or total samples (int, for MC methods).
- **params** (optional) — Array shaped `(batch, num_params)` for batched evaluation.
- **boundary** (optional) — Mask function to restrict the integration domain.

For full auto-generated module documentation, see the [package index](apidocs/index).
