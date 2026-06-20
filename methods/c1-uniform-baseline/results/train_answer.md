The task is to certify an upper bound on the first autocorrelation constant `C1` of Barnard and Steinerberger — the largest constant with `max_{|t|≤1/2}(f*f)(t) ≥ C1·(∫f)²` for every non-negative `f` supported on `[-1/4,1/4]`. Because `C1` is the infimum of `R(f) = max(f*f)/(∫f)²`, any admissible `f` I can write down certifies `C1 ≤ R(f)`, and lower is better. Before optimizing anything I want to nail down the trivial ceiling so I understand the functional and have a baseline that every later construction must beat, and so I can confirm the evaluator agrees with my own arithmetic.

The simplest legal function is the flat one — all heights equal, the discretized indicator of an interval. Working in the step-function class `f = Σ a_n·1_[n,n+1)`, the autoconvolution is piecewise linear and its peak sits at a node, namely `max_k(a*a)_k`. For `a_n = 1` over `N` pieces the discrete self-convolution is the triangular sequence `1,2,…,N,…,2,1` peaking at `N`, and the mass is `Σ a_n = N`, so the normalized score `R = 2N·max(a*a)/(Σa)² = 2N·N/N² = 2` exactly. Equivalently in the continuous picture, the indicator of width `w` has autoconvolution peak `h²w` and mass `hw`, giving `peak/(∫f)² = 1/w`, which the `2N` normalization carries to the constant `2` on the fixed interval. The number is independent of how finely I discretize: a flat vector of any length scores `2`. Piece count alone is not a lever — only the shape of the heights moves `R`.

The method here is just that flat profile, used deliberately as the ceiling. It is the unique maximally-symmetric member of the class, and crucially it is the *worst* case for this minimization: a flat function maximizes the central self-overlap, lining up every piece against every piece near zero shift, which is exactly what makes the autoconvolution's peak large relative to the mass. So it has no internal degree of freedom to spend, and every later gain must come from breaking this symmetry — introducing an asymmetric, structured, peak-suppressing height profile that pushes `max_k(a*a)_k` down while holding the mass fixed. The flat baseline buys none of that distance; it only establishes the starting altitude at `R = 2`, against which the descent toward the published frontier near `1.50286` (and the provable floor `1.28`) is measured.

It also doubles as a sanity check on the harness. The hand-computed triangle value `2` must match the evaluator exactly, and the published record sequence must score `1.5028628969` through the same `fftconvolve`-based evaluator — both of which I confirm, so I can trust the machine on the harder rungs.

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

def construct(N: int = 600):
    """Flat profile = discretized indicator; autoconvolution is a triangle, R = 2 exactly."""
    return [1.0] * N

if __name__ == "__main__":
    print(autoconv_c1_ratio(construct()))   # 2.0000000000000004
```
