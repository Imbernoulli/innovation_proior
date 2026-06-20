# Context: minimax Sequential-LP refinement for the C1 autocorrelation inequality

## Research question

Certify a tight *upper* bound on the first autocorrelation constant `C1` of Barnard–Steinerberger
(arXiv:1903.08731): `C1 = inf_f R(f)`, `R(f) = max_{|t|≤1/2}(f*f)(t)/(∫_{-1/4}^{1/4} f)²`, over non-negative `f`
supported on `[-1/4,1/4]`. Every admissible `f` certifies `C1 ≤ R(f)`; **lower is better**. Provable range
`C1 ∈ [1.28, 1.5028…]`.

## Construction class and score

Non-negative step function `f = Σ a_n·1_[n,n+1)`; autoconvolution piecewise linear with node values `b = a*a`,
peak `max_k b_k`; score

```
R(a) = 2N · max_k (a*a)_k / ( Σ_n a_n )^2 .
```

## This rung's obstruction

A coarse anneal (previous rung) reaches `1.5371` on `N=50` and then stalls — both from resolution and from a
deeper issue: when the peak is pushed down it flattens a whole *plateau* of near-equal peak nodes, and a softmax
gradient (or single-argmax subgradient) can only press down one node at a time, so it plateaus near `1.52`. The
true objective is the *maximum over the whole near-tight set* — a **minimax**.

## Method

Lift to `N=600` and solve the minimax by **Sequential Linear Programming**: epigraph variable `z` for the peak,
minimize `z` s.t. every node `(a*a)_k ≤ z` and mass fixed; linearize the quadratic nodes around the current `a`
(`∂(a*a)_k/∂a_j = 2 a_{k−j}`), solve the LP for the best peak-lowering step inside a small trust region, accept only
if the true `R` drops, iterate. Focus the LP on the top-K near-tight nodes (cheaper, faithful under a small trust),
with periodic full passes. This is the TTT-Discover / AutoEvolver "LP on the tight constraints" recipe. A
`β`-annealed Adam pass on a softmax-max surrogate provides the warm start.

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
