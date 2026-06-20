# Context: endpoint frontier for the C1 autocorrelation inequality

## Research question

Certify a tight *upper* bound on the first autocorrelation constant `C1` of Barnard–Steinerberger
(arXiv:1903.08731): `C1 = inf_f R(f)`, `R(f) = max_{|t|≤1/2}(f*f)(t)/(∫_{-1/4}^{1/4} f)²`, non-negative `f` on
`[-1/4,1/4]`. Every admissible `f` certifies `C1 ≤ R(f)`; **lower is better**. Provable range `C1 ∈ [1.28, 1.5028…]`.

## Construction class and score

Non-negative step function `f = Σ a_n·1_[n,n+1)`; piecewise-linear autoconvolution with node values `b = a*a`,
peak `max_k b_k`; score `R(a) = 2N · max_k (a*a)_k / (Σ_n a_n)^2`.

## This rung

The previous rung's trust-region Sequential-LP minimax converged to `1.5172` from one warm start at `N=600`. This
endpoint pushes toward the record `1.5028628969` (AutoEvolver / Claude-Opus, 30000 pieces; after TTT-Discover's
30000-piece `1.5028628983` and AlphaEvolve's 600-piece `1.5053`) by spending the two levers an agentic search would:
**basin diversity** (launch the SLP from several structurally different starts — the rung-3 shape plus boundary-spike
profiles, the asymmetric family the record constructions inhabit — and keep the global best) and a **long polish**
(a dedicated full-constraint SLP grind with restart kicks). The endpoint informed by the AutoEvolver 30000-piece
construction (boundary spikes over an irregular plateau) and its `fftconvolve` evaluator speedup.

## Fixed yardsticks

| Reference point | pieces `N` | `R` |
|---|---|---|
| Flat indicator (ceiling) | any | 2.0 |
| AlphaEvolve (App. B.1) | 600 | 1.5053 |
| TTT-Discover | 30000 | 1.5028628983 |
| AutoEvolver record | 30000 | 1.5028628969 |
| Provable lower bound | — | 1.28 |

The endpoint stops at the frontier a single bounded SLP constructor reaches (`~1.5170` at `N=600`), with the record
`1.5028628969` standing above as the still-open distance, since closing it needs a large-scale agentic/evolutionary
search, not a single local constructor.

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
