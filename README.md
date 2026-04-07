# FlashQuad

A user-friendly Python numerical integration library with GPU acceleration.

## Installation

We recommend using [uv](https://docs.astral.sh/uv/):
```bash
uv add flashquad
uv sync
```
or directly install:

```bash
uv pip install flashquad
```

## Quick start

```python
import numpy as np
from flashquad import FlashQuad

fq = FlashQuad(backend="numpy")

result = fq.simpson(
    func=lambda x, y: np.sin(x) * np.cos(y),
    intervals=[[0, np.pi], [0, np.pi]],
    num_points=[101, 101],
)
```

Switch to GPU by changing the backend (we recommend using CuPy for minimal setup):

```python
fq = FlashQuad(backend="cupy")
```

Batch thousands of parameter sets in a single GPU call:

```python
import cupy as cp
from flashquad import FlashQuad

fq = FlashQuad(backend="cupy")
params = cp.linspace(0.1, 10.0, 5000).reshape(-1, 1)  # 5000 parameter sets

results = fq.simpson(
    func=lambda x, a: cp.exp(-a * x**2),
    intervals=[[0, 5]],
    num_points=[201],
    params=params,
)
```

## Methods

| Method | Call |
|--------|------|
| Trapezoidal | `fq.trapz(...)` |
| Simpson's 1/3 | `fq.simpson(...)` |
| Boole's | `fq.booles(...)` |
| Gauss-Legendre | `fq.gauss(...)` |
| Monte Carlo | `fq.mc(...)` |
| Adaptive Monte Carlo | `fq.adpmc(...)` |

## Backends

NumPy, PyTorch, CuPy, and JAX.

## License

MIT
