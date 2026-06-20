# Context: coarse optimized step function for the C1 autocorrelation inequality

## Research question

Certify a tight *upper* bound on the first autocorrelation constant `C1` of Barnard–Steinerberger
(arXiv:1903.08731): the largest constant with `max_{|t|≤1/2}(f*f)(t) ≥ C1·(∫_{-1/4}^{1/4} f)²` for all
non-negative `f` supported on `[-1/4,1/4]`. Since `C1 = inf_f R(f)`, every admissible `f` certifies
`C1 ≤ R(f) = max(f*f)/(∫f)²`, so this is a **minimization** — lower is better. Provable range `C1 ∈ [1.28, 1.5028…]`.

## Construction class and score

Non-negative step function `f = Σ a_n·1_[n,n+1)`. The autoconvolution is piecewise linear with node values
`b = a*a` (discrete self-convolution), peaking at `max_k b_k`; the score is

```
R(a) = 2N · max_k (a*a)_k / ( Σ_n a_n )^2 ,    lower = tighter.
```

The flat profile (the previous baseline rung) is pinned at `R = 2` with no gradient — every gain must come from
breaking that symmetry, since a flat function maximizes the central self-overlap that this minimization punishes.

## This rung

Find the optimized *shape* at a coarse resolution `N ≈ 50`, short enough for a stochastic search to canvas the
shape space. The method: simulated annealing on the heights (multiplicative kicks, Metropolis acceptance that takes
occasional uphill moves to escape the flat-triangle basin, geometric cooling, several restarts), then a
`β`-annealed projected-Adam polish on a softmax surrogate of the non-differentiable `max`. The fine-grid refinement
that a much stronger optimizer would bring is deferred to later rungs.

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
