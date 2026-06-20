# Context: the first autocorrelation inequality C1 (upper-bound minimization)

## Research question

The **first autocorrelation inequality** of Barnard and Steinerberger (arXiv:1903.08731) asks for the
largest constant `C1` such that

```
max_{|t|≤1/2} (f*f)(t) ≥ C1 · ( ∫_{-1/4}^{1/4} f )^2     for ALL non-negative f supported on [-1/4,1/4],
```

where `(f*f)(t) = ∫ f(s) f(t−s) ds` is the autoconvolution. Equivalently `C1 = inf_f R(f)` over admissible
`f`, with

```
R(f) = max_{|t|≤1/2} (f*f)(t) / ( ∫_{-1/4}^{1/4} f )^2 .
```

Every admissible `f` with a measured `R(f)` is a *constructive upper bound* `C1 ≤ R(f)`, so this is a
**minimization**: lower `R` is a tighter, better certificate. This is the opposite direction from the second
inequality `C2` (a maximization). The provable lower bound is `C1 ≥ 1.28` (Cloninger–Steinerberger, 2017),
so `C1 ∈ [1.28, 1.5028…]`.

## Construction class and score

A candidate is a non-negative **piecewise-constant step function** `f = Σ_{n=0}^{N−1} a_n·1_[n,n+1)`,
`a_n ≥ 0`, `N` unit-width pieces. The objective is translation- and dilation-invariant, so only the heights
and their count matter. The autoconvolution of a step function is **piecewise linear**: its node values are
the discrete self-convolution `b = a*a`, `b_k = Σ_n a_n a_{k−n}`, and the maximum over `t` equals `max_k b_k`
(a piecewise-linear curve peaks at a node). The mass is `Σ_n a_n`. The scale-normalized score every modern
result uses is

```
R(a) = 2N · max_k (a*a)_k / ( Σ_n a_n )^2 .
```

The `2N` factor carries the dimensionless discrete ratio to the continuous constant on `[-1/4,1/4]`; it makes
the flat indicator score exactly `2`.

## Fixed yardsticks

| Reference point | pieces `N` | `R` (upper bound on `C1`) |
|---|---|---|
| Flat indicator (ceiling) | any | 2.0 |
| Previous published record (pre-2025) | — | 1.5098 |
| AlphaEvolve (App. B.1) | 600 | 1.5053 |
| TTT-Discover | 30000 | 1.5028628983 |
| AutoEvolver record (Claude/Opus) | 30000 | 1.5028628969 |
| Provable lower bound (Cloninger–Steinerberger) | — | 1.28 |

## Evaluator (frozen)

```python
import numpy as np
from scipy.signal import fftconvolve

def autoconv_c1_ratio(a):
    """R = 2N * max(a*a) / (sum a)^2 for f = sum_n a_n 1_[n,n+1). Lower = tighter upper bound on C1."""
    a = np.clip(np.asarray(a, dtype=float), 0.0, 1000.0)
    N = len(a)
    b = fftconvolve(a, a)
    peak = float(np.max(b))
    s = float(np.sum(a))
    if s < 0.01:
        return float("inf")
    return 2.0 * N * peak / (s * s)
```

This baseline rung fixes the trivial ceiling and validates the evaluator (flat = `2` exactly; the AutoEvolver
record sequence = `1.5028628969` exactly) before any optimization begins.
