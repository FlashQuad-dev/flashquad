# FlashQuad

**A user-friendly Python numerical integration library with GPU acceleration.**

FlashQuad provides a single unified API for numerical integration across multiple
array backends — NumPy, PyTorch, CuPy, and JAX — so you can move from CPU
prototyping to GPU-accelerated computation with one line change.

## Key features

- **Multiple quadrature methods** — trapezoidal, Simpson's, Boole's, Gauss-Legendre, Monte Carlo, and adaptive Monte Carlo
- **GPU acceleration** — run on CUDA via PyTorch, CuPy, or JAX with no code changes beyond swapping the backend
- **Batched parameter sweeps** — evaluate integrals over a batch of parameters in a single call
- **Arbitrary dimensions** — integrate over 1D, 2D, 3D, or higher-dimensional domains
- **Boundary masking** — apply custom domain boundaries via mask functions

## Quick example

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

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started
api
apidocs/index
```
