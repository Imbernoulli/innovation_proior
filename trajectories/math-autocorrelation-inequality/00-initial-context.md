## Research question

Among non-negative functions `f` on the real line, how large can the ratio

```
R(f) = ||f*f||_2^2 / ( ||f*f||_inf · ||f*f||_1 )
```

be made, where `f*f` is the autoconvolution `(f*f)(x) = ∫ f(t) f(x−t) dt`? Hölder's inequality gives `R(f) ≤ 1` for any function, with equality only when `f*f` is an indicator; an autoconvolution is always a smooth bump, never an indicator, so the supremum is strictly below 1. The problem of finding `C2 := sup_f R(f)` is the **second autocorrelation inequality** of Barnard and Steinerberger (arXiv:1903.08731). The design task is to produce one concrete non-negative function `f` with as large a value of `R(f)` as possible.

The standard construction class is a non-negative **piecewise-constant step function**

```
f = Σ_{n=0}^{N−1} v_n · 1_[n, n+1),   v_n ≥ 0,
```

with `N` pieces of unit width and heights `v_n`. The objective is invariant under translation and dilation of `f`, so only the heights and their count matter; the autoconvolution of a step function is piecewise linear and fully determined by its values at the integer nodes.

## Prior art / Background / Baselines

Write `L_j = (f*f)(j)` for the node values, `0 ≤ j ≤ 2N`. These are the discrete self-convolution of the height vector: `L_j = c_{j−1}` where `c = v * v`. On each unit cell `[j, j+1)` the autoconvolution is the straight line from `L_j` to `L_{j+1}`, so the three norms are exact integrals:

```
||f*f||_inf = max_j L_j
||f*f||_1   = ½ Σ_j (L_j + L_{j+1})
||f*f||_2^2 = ⅓ Σ_j (L_j^2 + L_j·L_{j+1} + L_{j+1}^2)
```

The score is `R = ||f*f||_2^2 / (||f*f||_inf · ||f*f||_1)`, a deterministic functional bounded by 1.

Published lower bounds for `C2`:

| Baseline | steps `N` | `R` |
|---|---|---|
| Flat indicator (uniform heights) | any | 0.6667 |
| Matolcsi–Vinuesa | 20 | 0.88922 |
| AlphaEvolve | 50 | 0.89628 |
| Boyer–Li | 575 | 0.901564 |
| Jaech–Joseph | 539 | ~0.9016 |
| AlphaEvolve-V2 | ~50000 | 0.96102 |

Each baseline leaves a visible gap:

- **Flat indicator.** A single uniform height produces a triangular autoconvolution, the lowest meaningful score.
- **Matolcsi–Vinuesa.** A small hand-tuned step profile reaches `0.88922`, but the construction does not generalize to larger `N`.
- **AlphaEvolve.** A learned 50-step shape improves on hand tuning, yet the score is still far below 1.
- **Boyer–Li.** Simulated annealing plus gradient refinement on 575 steps reaches `~0.9016`, but only after roughly `10^6` gradient trajectories.
- **Jaech–Joseph.** Independent search on 539 steps matches `~0.9016`, confirming the difficulty of pushing past that regime rather than offering a transferable design.
- **AlphaEvolve-V2.** An evolutionary search on `~50000` irregular steps reaches `0.96102`, the highest published score, at an enormous search cost and with no compact description of the resulting function.

The gap between `0.96102` and the Hölder ceiling `1` remains open.

## Fixed substrate / Code framework

The harness receives a height vector `v`, clips any negative entries to `0`, forms the autoconvolution nodes `L` by self-convolution, computes the three piecewise-linear norms by the exact formulas above, and returns `R`. Self-convolution uses an FFT (`scipy.signal.fftconvolve`), so the evaluator is `O(N log N)` and scales to tens of thousands of pieces. The norm conventions and the score definition are frozen.

```python
import numpy as np
from scipy.signal import fftconvolve

def autoconv_ratio(v):
    """R = ||f*f||_2^2 / (||f*f||_inf * ||f*f||_1) for f = sum_n v_n 1_[n,n+1)."""
    v = np.clip(np.asarray(v, dtype=float), 0.0, None)
    N = len(v)
    c = fftconvolve(v, v)                 # length 2N-1: c[k] = sum_n v_n v_{k-n}
    L = np.zeros(2 * N + 1)               # node values L_j = (f*f)(j), L_0 = L_2N = 0
    L[1:2 * N] = c                        # L_j = c_{j-1}
    Lj, Ljp = L[:-1], L[1:]               # consecutive node pairs, one per unit cell
    l2sq = (1.0 / 3.0) * np.sum(Lj**2 + Lj * Ljp + Ljp**2)   # ||f*f||_2^2
    l1   = 0.5 * np.sum(Lj + Ljp)                            # ||f*f||_1
    linf = np.max(L)                                         # ||f*f||_inf
    return float(l2sq / (linf * l1))
```

## Editable interface

The editable output is a list or one-dimensional array of non-negative finite floats `v = [v_0, ..., v_{N−1}]` with arbitrary length `N ≥ 1`. The harness clips negatives and computes `R(v)`; no other constraints or side channels are available. The constructor may search, optimize, or hand-design the heights, but the final returned object must be the height vector itself.

## Evaluation settings

A single deterministic functional `R(v)`. The harness reports the ratio of the returned vector; any internal randomness is fixed to a stated seed so the reported number is reproducible. The fixed yardsticks are the Hölder ceiling `1` and the published lower bounds `0.6667`, `0.88922`, `0.89628`, `0.901564`/`~0.9016`, and `0.96102`. The score is the entire result; there is no partial credit beyond the ratio.
