# Context: refining the C1 autocorrelation construction at high resolution

## Research question

Certify a tight *upper* bound on the first autocorrelation constant `C1` of Barnard–Steinerberger
(arXiv:1903.08731): `C1 = inf_f R(f)`, `R(f) = max_{|t|≤1/2}(f*f)(t)/(∫_{-1/4}^{1/4} f)²`, over non-negative `f`
supported on `[-1/4,1/4]`. Every admissible `f` certifies `C1 ≤ R(f)`; **lower is better**. Provable range
`C1 ∈ [1.28, 1.5028…]`.

The task at this rung: starting from a coarse warm-start profile, refine the discretized construction at high
resolution to push `R` further down.

## Construction class and score

Non-negative step function `f = Σ a_n·1_[n,n+1)`; autoconvolution piecewise linear with node values `b = a*a`,
peak `max_k b_k`; score

```
R(a) = 2N · max_k (a*a)_k / ( Σ_n a_n )^2 .
```

The objective is the maximum over all autoconvolution nodes. The gradient of a self-convolution node with respect
to a height is a shifted copy of the heights, `∂(a*a)_k/∂a_j = 2 a_{k−j}`.

## Warm start

A `β`-annealed Adam pass on a softmax-max surrogate of the objective brings a flat initialization down to about
`1.52` on a moderate grid, and provides the starting profile for this rung.

## Fixed yardsticks

| Reference point | pieces `N` | `R` |
|---|---|---|
| Flat indicator (ceiling) | any | 2.0 |
| AlphaEvolve (App. B.1) | 600 | 1.5053 |
| TTT-Discover | 30000 | 1.5028628983 |
| AutoEvolver record | 30000 | 1.5028628969 |
| Provable lower bound | — | 1.28 |

## Evaluator (frozen)

```python
import numpy as np
from scipy.signal import fftconvolve

def autoconv_c1_ratio(a):
    a = np.clip(np.asarray(a, dtype=float), 0.0, 1000.0)
    N = len(a)
    b = fftconvolve(a, a)
    s = float(np.sum(a))
    if s < 0.01:
        return float("inf")
    return 2.0 * N * float(np.max(b)) / (s * s)
```
