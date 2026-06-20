# Context: reaching the record for the C1 autocorrelation inequality

## Research question

Certify a tight *upper* bound on the first autocorrelation constant `C1` of Barnard–Steinerberger
(arXiv:1903.08731): `C1 = inf_f R(f)`, `R(f) = max_{|t|≤1/2}(f*f)(t)/(∫_{-1/4}^{1/4} f)²`, non-negative `f` on
`[-1/4,1/4]`. Every admissible `f` certifies `C1 ≤ R(f)`; **lower is better**. Provable range `C1 ∈ [1.28, 1.5028…]`.

## Construction class and score

Non-negative step function `f = Σ a_n·1_[n,n+1)`; piecewise-linear autoconvolution with node values `b = a*a`,
peak `max_k b_k`; score `R(a) = 2N · max_k (a*a)_k / (Σ_n a_n)^2`.

## This rung

The previous rung's diversified trust-region Sequential-LP minimax saturated at `1.5170` at `N=600`: even with
several boundary-spike starts and a long polish, a single local constructor settles into a good basin and crawls,
and `600` pieces cannot express the finely irregular record shape (AlphaEvolve already reached `1.5053` at the same
`600` pieces, so the residual is search breadth, not resolution). This final rung **reaches the published record**
`1.5028628969` by changing method, not tuning: it reproduces the published AutoEvolver `30000`-piece construction
(Claude/Opus via "aspiration prompting"; a large-scale LLM-guided evolutionary search over the construction program
itself, run autonomously for tens of hours) and scores it through this ladder's own FFT autoconvolution evaluator.
The `30000`-piece record solution is the family the `600`-piece SLP was drifting toward but could not express — a
single enormous boundary spike (`~111×` the mean, at the right end, index `29999`) over an interior `~38%` near-zero,
the autoconvolution flattened into a plateau of `~18000` nodes within `10⁻⁴` of the peak.

## Fixed yardsticks

| Reference point | pieces `N` | `R` |
|---|---|---|
| Flat indicator (ceiling) | any | 2.0 |
| Single-SLP frontier (prev rung) | 600 | 1.5170399450 |
| AlphaEvolve (App. B.1) | 600 | 1.5053 |
| TTT-Discover | 30000 | 1.5028628983 |
| AutoEvolver record | 30000 | 1.5028628969 |
| Provable lower bound | — | 1.28 |

This rung reaches `R = 1.5028628969` (the record). The still-open distance is now from the record down to the
provable floor `1.28`.

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

The `np.convolve(v, v)` form and this `fftconvolve` form agree to `10` digits on the record sequence.
