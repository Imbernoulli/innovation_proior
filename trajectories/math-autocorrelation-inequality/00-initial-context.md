## Research question

Among all non-negative functions `f` on the real line, how large can the ratio

```
R(f) = ||f*f||_2^2 / ( ||f*f||_inf · ||f*f||_1 )
```

be made, where `f*f` is the autoconvolution `(f*f)(x) = ∫ f(t) f(x−t) dt`? Hölder's inequality
gives `R(f) ≤ 1` for *any* function, with equality only when `f*f` is (a scalar multiple of) an
indicator — but the autoconvolution of a non-negative function is a smooth, bump-shaped object, never
an indicator, so the trivial bound is not attained and the real question is the supremum `C2 :=
sup_f R(f)`. This is the **second autocorrelation inequality** of Barnard and Steinerberger
(arXiv:1903.08731); they asked whether `C2` can be pushed close to `1`, and every published advance
since has been a *constructive lower bound* — an explicit `f` with a measured `R(f)`. The single thing
being designed here is exactly that: a constructor that emits one concrete non-negative function, scored
by `R(f)` alone, higher being better.

The construction class is fixed and is the one used by every modern result on this problem. A candidate
is a non-negative **piecewise-constant step function**

```
f = Σ_{n=0}^{N−1} v_n · 1_[n, n+1),   v_n ≥ 0,
```

`N` pieces of unit width with heights `v_n`. The objective is invariant under translation and dilation
of `f`, so only the heights and their count matter — the grid spacing and offset wash out. This is what
makes the problem cleanly computable: the autoconvolution of a step function is **piecewise linear**, so
it is fully determined by its values at the integer nodes.

## How the score is defined

Write `L_j = (f*f)(j)` for the node values of the autoconvolution, `0 ≤ j ≤ 2N`. Because `f` is constant
on each unit interval, the overlap integral collapses to a discrete autocorrelation of the heights:

```
L_j = Σ_n v_n · v_{j−n−1}     (sum over the valid overlap range),   L_0 = L_{2N} = 0.
```

Equivalently `L_j = c_{j−1}` where `c = v * v` is the length-`(2N−1)` discrete convolution of the height
vector with itself. On each unit cell `[j, j+1)` the autoconvolution is the straight line from `L_j` to
`L_{j+1}`, and the three norms are exact integrals of that piecewise-linear curve:

```
||f*f||_inf = max_j L_j
||f*f||_1   = ½ Σ_j (L_j + L_{j+1})                       (trapezoid areas)
||f*f||_2^2 = ⅓ Σ_j (L_j^2 + L_j·L_{j+1} + L_{j+1}^2)     (∫ of a linear segment squared)
```

and the score is `R = ||f*f||_2^2 / (||f*f||_inf · ||f*f||_1)`. There is no held-out set and no way to
game the metric: the number is a deterministic functional of the heights, and `R ≤ 1` always.

A few fixed yardsticks anchor every rung. A **flat** step function (the discretized indicator) has a
triangular autoconvolution and scores exactly `2/3 ≈ 0.6667` — the floor. Matolcsi and Vinuesa found a
`20`-step function reaching `0.88922` (Canad. Math. Bull., 2024). Google DeepMind's **AlphaEvolve** gave
a `50`-step function at `0.89628` (arXiv:2506.13131, App. B.2). **Boyer & Li** pushed a `575`-step
function to `0.901564` with simulated annealing plus gradient refinement (arXiv:2506.16750); Jaech &
Joseph reached `~0.9016` independently with a `539`-step function (arXiv:2508.02803). The current record
is **AlphaEvolve-V2** at `C2 ≥ 0.96102`, a deliberately irregular `~50000`-step function. The headline
numbers to keep in view:

| Reference point | steps `N` | `R` |
|---|---|---|
| Hölder ceiling (provable, unattained) | — | 1.0 |
| **AlphaEvolve-V2 record** (irregular step fn) | ~50000 | **0.96102** |
| Boyer–Li / Jaech–Joseph | 575 / 539 | ~0.9016 |
| AlphaEvolve | 50 | 0.89628 |
| Matolcsi–Vinuesa | 20 | 0.88922 |
| Flat indicator (this scaffold's floor) | any | 0.6667 |

The record has been climbed by dedicated optimization spending enormous compute (Boyer–Li ran `~10^6`
gradient trajectories; AlphaEvolve-V2 is an evolutionary search). Reproducing `0.96102` from scratch in a
single bounded constructor is not expected — so the ladder here is not chasing a known-easy target, it is
climbing from the trivial flat floor toward the published frontier, and the gap to `0.96102` is the
honest measure of how open the problem still is.

## The related constants (for orientation)

This task focuses on `C2` because it is cleanly runnable, but it sits in a small family. The **first**
autocorrelation inequality (`C1`) is a *minimization*: the smallest constant `c` with `max_x (f*f)(x) ≥
c·(∫f)^2 / supp(f)` over non-negative `f` supported on an interval — an *upper* bound problem where
*lower* is better; the record there is `C1 ≤ 1.5028628969`, held by Claude Code / Opus via "aspiration
prompting" in AutoEvolver (beating TTT-Discover `1.5028628983` and AlphaEvolve `1.5053`). The **third**
(`C3`) is a related ratio variant. Only `C2` is built here; `C1`/`C3` are cited as the surrounding
landscape.

## The fixed substrate

The harness is a thin, deterministic evaluator. It receives a height vector `v` (a Python list of
non-negative floats), clips negatives to `0`, forms the autoconvolution nodes `L` by self-convolution,
computes the three piecewise-linear norms by the exact formulas above, and returns `R`. Self-convolution
is done with an FFT (`scipy.signal.fftconvolve`), so the evaluator is `O(N log N)` and scales to tens of
thousands of pieces. The formulas, the norm conventions, and `R ≤ 1` are frozen.

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

Every valid candidate is a list of non-negative finite floats of any length `N ≥ 1`. There are no other
constraints — the constructor is free to return any height vector, structured, optimized, or searched.

## Evaluation settings

A single deterministic functional `R(v)` above. Because a constructor may search internally, the harness
reports `R` of the *returned* vector and fixes any stochastic run to a stated seed so the number is
reproducible. The fixed yardsticks — flat floor `0.6667`, Matolcsi–Vinuesa `0.88922`, AlphaEvolve
`0.89628`, Boyer–Li `~0.9016`, AlphaEvolve-V2 record `0.96102` — are the rulers every rung is read
against. The score is the whole result; there is no partial credit beyond the ratio itself.
