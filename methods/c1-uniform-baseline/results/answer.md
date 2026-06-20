**Problem.** Construct a non-negative step function `f = Σ a_n·1_[n,n+1)` *minimizing* `R = 2N·max_k(a*a)_k /
(Σ a_n)^2`, the normalized peak of the piecewise-linear autoconvolution. Every admissible `f` certifies an upper
bound `C1 ≤ R(f)` on the first autocorrelation constant of Barnard–Steinerberger; **lower is better**. The provable
floor is `C1 ≥ 1.28`; the published record upper bound is `1.5028628969`. This rung fixes the ceiling.

**Key idea.** The simplest legal candidate is the flat profile — all heights equal, i.e. the discretized indicator
of an interval. Its discrete self-convolution `a*a` is the triangular sequence `1,2,…,N,…,2,1`, peaking at the
center value `N`; the mass is `Σ a_n = N`. Hence `R = 2N·N / N^2 = 2` exactly. Equivalently, the continuous
indicator of width `w` has autoconvolution peak `h^2 w` and mass `h w`, so `peak/(∫f)^2 = 1/w`, which the `2N`
normalization carries to the constant `2` on the fixed interval. This value is independent of the number of pieces
`N` — refining a flat profile leaves the triangle, and the value `2`, unchanged — so piece count alone is not a
lever; only the *shape* of the heights moves `R`.

**Why these choices.** The flat function is the unique maximally-symmetric member of the class and has no internal
degree of freedom: it maximizes the central self-overlap (the worst case for this minimization), so every later
gain must come from suppressing the autoconvolution's single dominant peak by introducing an asymmetric, structured
height profile. It is deliberately the ceiling — rigid, parameter-free, guaranteed legal — and it doubles as a
sanity check on the harness, since the hand-computed triangle value `2` must match the evaluator exactly, and the
published record sequence must score `1.50286` through the same evaluator. The entire distance from `2.0` down to
the frontier near `1.50286` is what later, searched rungs must buy.

**Hyperparameters / contract.** None. Output is a flat non-negative height vector of length `N` (default `N = 600`;
any `N ≥ 1` gives the same `R`). Deterministic — same vector every call, `R = 2` exactly.

```python
import numpy as np
from scipy.signal import fftconvolve

def autoconv_c1_ratio(a):
    a = np.clip(np.asarray(a, dtype=float), 0.0, 1000.0)
    N = len(a)
    b = fftconvolve(a, a)
    peak = float(np.max(b))
    s = float(np.sum(a))
    if s < 0.01:
        return float("inf")
    return 2.0 * N * peak / (s * s)

def construct(N: int = 600):
    """Flat profile = discretized indicator; autoconvolution is a triangle, R = 2 exactly."""
    return [1.0] * N

if __name__ == "__main__":
    print(autoconv_c1_ratio(construct()))   # 2.0000000000000004
```
