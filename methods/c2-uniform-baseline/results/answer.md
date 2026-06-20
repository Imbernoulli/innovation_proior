**Problem.** Construct a non-negative step function `f = Σ v_n·1_[n,n+1)` maximizing `R = ||f*f||_2^2 /
(||f*f||_inf·||f*f||_1)`, where the autoconvolution `f*f` is piecewise linear with node values `L_j =
(f*f)(j)`. Hölder gives `R ≤ 1`, unattained because an autoconvolution is never an indicator; the true
supremum `C2` is the open quantity, with published lower bounds up to `0.96102`. This rung fixes the floor.

**Key idea.** The simplest legal candidate is the flat profile — all heights equal, i.e. the discretized
indicator of an interval. Its autoconvolution is the triangle ("tent") on `[0,2]`: peak `1`, so
`||f*f||_inf = 1`; area `½·base·height = 1`, so `||f*f||_1 = 1`; and `||f*f||_2^2 = 2∫_0^1 x^2 dx = 2/3`.
Hence `R = (2/3)/(1·1) = 2/3` exactly. This value is independent of the number of pieces `N` — refining a
flat profile leaves the triangle unchanged — so piece count alone is not a lever; only the *shape* of the
heights moves `R`.

**Why these choices.** The flat function is the unique maximally-symmetric member of the class and has no
internal degree of freedom: every later gain must come from bending the autoconvolution away from the tent
(flatter cap, steeper sides, toward the indicator Hölder rewards). It is deliberately the floor — rigid,
parameter-free, guaranteed legal — and it doubles as a sanity check on the harness, since the hand-computed
triangle value `2/3` must match the evaluator exactly. The entire distance from `0.6667` to the frontier
near `0.96` is what later, searched rungs must buy by introducing asymmetric, structured heights.

**Hyperparameters / contract.** None. Output is a flat non-negative height vector of length `N` (default
`N = 50`; any `N ≥ 1` gives the same `R`). Deterministic — same vector every call, `R = 2/3` exactly.

```python
import numpy as np
from scipy.signal import fftconvolve

def autoconv_ratio(v):
    v = np.clip(np.asarray(v, dtype=float), 0.0, None)
    N = len(v)
    c = fftconvolve(v, v)
    L = np.zeros(2 * N + 1)
    L[1:2 * N] = c
    Lj, Ljp = L[:-1], L[1:]
    l2sq = (1.0 / 3.0) * np.sum(Lj**2 + Lj * Ljp + Ljp**2)
    l1   = 0.5 * np.sum(Lj + Ljp)
    linf = np.max(L)
    return float(l2sq / (linf * l1))

def construct(N: int = 50):
    """Flat profile = discretized indicator; autoconvolution is a triangle, R = 2/3 exactly."""
    return [1.0] * N

if __name__ == "__main__":
    print(autoconv_ratio(construct()))   # 0.6666666666666666
```
