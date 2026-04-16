"""Random smooth integrands for fuzz testing.

Each :class:`RandomIntegrand` is a finite sum of bounded primitive terms
(polynomial, sine, cosine, Gaussian bump). It is callable with either Python
floats (as ``scipy.integrate.nquad`` passes them), numpy arrays (flashquad's
numpy backend), or torch tensors (torchquad). The namespace is inferred at
call time via ``array_api_compat``.
"""

import numpy as np

try:
    import array_api_compat
except ImportError:
    array_api_compat = None


def _namespace(x):
    """Return an array namespace usable for *x* — falls back to numpy for scalars."""
    if array_api_compat is not None:
        try:
            return array_api_compat.array_namespace(x)
        except Exception:
            pass
    return np


_KINDS = ("poly", "sin", "cos", "gauss")


class RandomIntegrand:
    """A randomly-generated, smooth, bounded ``dim``-variate function.

    Parameters
    ----------
    dim : int
        Number of input variables.
    rng : numpy.random.Generator
        Source of randomness.
    num_terms : int
        Number of primitive terms summed to form the integrand.
    """

    def __init__(self, dim: int, rng: np.random.Generator, num_terms: int = 3):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self.num_terms = num_terms
        self.terms = [self._random_term(rng) for _ in range(num_terms)]

    def _random_term(self, rng):
        kind = rng.choice(_KINDS)
        coef = float(rng.uniform(-1.0, 1.0))
        if kind == "poly":
            powers = rng.integers(0, 3, size=self.dim).tolist()
            return ("poly", coef, powers)
        if kind in ("sin", "cos"):
            freqs = rng.uniform(-1.5, 1.5, size=self.dim).tolist()
            return (kind, coef, freqs)
        # gauss bump
        scales = rng.uniform(0.2, 1.5, size=self.dim).tolist()
        center = rng.uniform(-0.5, 0.5, size=self.dim).tolist()
        return ("gauss", coef, scales, center)

    def __call__(self, *xs):
        if len(xs) != self.dim:
            raise ValueError(f"expected {self.dim} args, got {len(xs)}")
        xp = _namespace(xs[0])
        result = None
        for term in self.terms:
            piece = self._eval_term(term, xs, xp)
            result = piece if result is None else result + piece
        return result

    @staticmethod
    def _eval_term(term, xs, xp):
        kind = term[0]
        if kind == "poly":
            _, coef, powers = term
            val = coef
            for x, p in zip(xs, powers):
                if p == 0:
                    continue
                val = val * (x ** int(p))
            return val
        if kind == "sin":
            _, coef, freqs = term
            arg = sum(float(f) * x for f, x in zip(freqs, xs))
            return coef * xp.sin(arg)
        if kind == "cos":
            _, coef, freqs = term
            arg = sum(float(f) * x for f, x in zip(freqs, xs))
            return coef * xp.cos(arg)
        # gauss bump: coef * exp(-sum(scale_i * (x_i - c_i)^2))
        _, coef, scales, center = term
        arg = sum(
            float(s) * (x - float(c)) ** 2 for s, c, x in zip(scales, center, xs)
        )
        return coef * xp.exp(-arg)

    def describe(self) -> str:
        """Short human-readable signature used in fuzz reports."""
        parts = []
        for term in self.terms:
            kind = term[0]
            if kind == "poly":
                _, c, p = term
                parts.append(f"{c:+.3f}*poly(p={p})")
            elif kind in ("sin", "cos"):
                _, c, f = term
                parts.append(f"{c:+.3f}*{kind}(f={[round(v, 2) for v in f]})")
            else:
                _, c, s, ctr = term
                parts.append(
                    f"{c:+.3f}*gauss(s={[round(v, 2) for v in s]},"
                    f"c={[round(v, 2) for v in ctr]})"
                )
        return " + ".join(parts)


def random_intervals(dim: int, rng: np.random.Generator) -> list[list[float]]:
    """Random bounded integration intervals around the origin."""
    intervals = []
    for _ in range(dim):
        a = float(rng.uniform(-1.0, 0.2))
        b = float(rng.uniform(0.3, 1.2))
        lo, hi = min(a, b), max(a, b)
        intervals.append([lo, hi])
    return intervals
