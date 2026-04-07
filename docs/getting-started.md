# Getting started

## Installation

Install from PyPI:

```bash
pip install flashquad
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add flashquad
```

### GPU backends

FlashQuad works with any of these backends. Install the one you need:

| Backend | Install command | GPU support |
|---------|----------------|-------------|
| NumPy | included with flashquad | CPU only |
| PyTorch | `pip install torch` | CUDA |
| CuPy | `pip install cupy-cuda12x` | CUDA |
| JAX | `pip install jax[cuda12]` | CUDA |

## Basic usage

Every workflow starts by creating a `FlashQuad` instance bound to a backend:

```python
from flashquad import FlashQuad

fq = FlashQuad(backend="numpy")
```

Then call any integration method. They all share the same signature:

```python
result = fq.trapz(
    func=lambda x: x ** 2,
    intervals=[[0, 1]],
    num_points=[101],
)
```

### Switching to GPU

Just change the backend string:

```python
fq = FlashQuad(backend="torch")           # auto-selects CUDA if available
fq = FlashQuad(backend="torch", device="cpu")  # force CPU
```

### Multidimensional integrals

Pass multiple intervals and grid sizes:

```python
import numpy as np

fq = FlashQuad(backend="numpy")

result = fq.simpson(
    func=lambda x, y: np.sin(x) * np.cos(y),
    intervals=[[0, np.pi], [0, np.pi]],
    num_points=[101, 101],
)
```

### Batched parameter sweeps

Pass a `params` array to evaluate the same integral over many parameter sets at once:

```python
import numpy as np

fq = FlashQuad(backend="numpy")
params = np.array([[1.0], [2.0], [3.0]])  # 3 parameter sets

result = fq.trapz(
    func=lambda x, a: a * x ** 2,
    intervals=[[0, 1]],
    num_points=[101],
    params=params,
)
# result.shape == (3,)
```

### Boundary masking

Use the `boundary` argument to restrict integration to a sub-region:

```python
import numpy as np

fq = FlashQuad(backend="numpy")

result = fq.simpson(
    func=lambda x, y: 1.0,
    intervals=[[-1, 1], [-1, 1]],
    num_points=[101, 101],
    boundary=lambda x, y: x ** 2 + y ** 2 <= 1,  # unit disk
)
```

## Available methods

| Method | Call | Notes |
|--------|------|-------|
| Trapezoidal | `fq.trapz(...)` | Simple, general-purpose |
| Simpson's 1/3 | `fq.simpson(...)` | Higher order; `num_points` must be odd |
| Boole's | `fq.booles(...)` | Higher order; `(num_points - 1)` divisible by 4 |
| Gauss-Legendre | `fq.gauss(...)` | High accuracy for smooth functions |
| Monte Carlo | `fq.mc(...)` | `num_points` is total sample count (single int) |
| Adaptive Monte Carlo | `fq.adpmc(...)` | Iterative refinement; extra `num_iterations` arg |
