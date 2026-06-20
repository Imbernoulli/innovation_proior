## Research question

Among all non-negative functions `f` supported on `[-1/4, 1/4]`, how *small* can the ratio

```
R(f) = max_{|t|≤1/2} (f*f)(t) / ( ∫_{-1/4}^{1/4} f )^2
```

be made, where `f*f` is the autoconvolution `(f*f)(t) = ∫ f(s) f(t−s) ds`? The **first autocorrelation
inequality** of Barnard and Steinerberger (arXiv:1903.08731) asks for the largest constant `C1` with

```
max_{|t|≤1/2} (f*f)(t) ≥ C1 · ( ∫_{-1/4}^{1/4} f )^2     for ALL non-negative f supported on [-1/4,1/4].
```

`C1` is the infimum of `R(f)` over all admissible `f`. So unlike its sibling `C2` — which is a *maximization*
where higher `R` is better — this is a **minimization**: every admissible `f` with a measured `R(f)` is a
*constructive upper bound* `C1 ≤ R(f)`, and a *lower* `R` is a *tighter*, better certificate. The single thing
being designed here is exactly that: a constructor that emits one concrete non-negative function, scored by
`R(f)` alone, **lower being better**. This is the opposite direction from the `C2` ladder; do not carry over its
numbers or its "climb toward 1" intuition — here we are pushing a ceiling *down* toward an unknown floor.

The construction class is fixed and is the one used by every modern result on this problem. A candidate is a
non-negative **piecewise-constant step function**

```
f = Σ_{n=0}^{N−1} a_n · 1_[n, n+1),   a_n ≥ 0,
```

`N` pieces of unit width with heights `a_n`. The objective is invariant under translation and dilation of `f`,
so only the heights and their count matter — the grid spacing and offset wash out. This is what makes the problem
cleanly computable: the autoconvolution of a step function is **piecewise linear**, its node values are a discrete
self-convolution of the heights, and the maximum over `t` is attained at one of those nodes.

## How the score is defined

Let `b = a * a` be the length-`(2N−1)` discrete convolution of the height vector with itself, `b_k = Σ_n a_n
a_{k−n}`. Because `f` is constant on each unit interval, the continuous autoconvolution `f*f` is the piecewise-linear
interpolant through the values `b_k` at the nodes, so its maximum over all `t` equals `max_k b_k` exactly — the
peak of a piecewise-linear curve sits at a node. The integral `∫ f` is just `Σ_n a_n`. With the unit grid (width
`1`, so the support has length `N` and the admissible interval `[-1/4,1/4]` is the normalized version of `[0,N]`),
the scale-normalized score used by every modern result is

```
R(a) = 2 N · max_k (a*a)_k / ( Σ_n a_n )^2 .
```

The factor `2N` is the bookkeeping that turns the dimensionless discrete ratio into the continuous constant `C1`
on `[-1/4,1/4]`: it is what makes the *flat* indicator score exactly `2` (derived below), matching the trivial
continuous bound. There is no held-out set and no way to game the metric — the number is a deterministic functional
of the heights, and any admissible `f` certifies `C1 ≤ R(a)`.

A few fixed yardsticks anchor every rung. The **flat** step function (the discretized indicator) has a triangular
autoconvolution and scores exactly `2` — the trivial ceiling, the starting altitude. The provable *lower* bound on
`C1` is `1.28` (Cloninger–Steinerberger, 2017), so the true `C1` lives in `[1.28, 1.5028…]` and every construction
here is an upper certificate sitting above it. The published upper-bound record was `1.5098` for years; then **AlphaEvolve**
(arXiv:2506.13131, App. B.1) reached `1.5053` with a `600`-interval step function (a later variant reports `1.5032`);
**TTT-Discover** (arXiv:2601.16175) pushed to `1.5028628983` with a `30000`-piece construction; and **AutoEvolver**
(Claude/Opus, via "aspiration prompting"; https://github.com/tengxiaoliu/autoevolver,
https://tengxiaoliu.github.io/autoevolver/) holds the current record `1.5028628969`, a `30000`-piece step function.
The headline numbers to keep in view:

| Reference point | pieces `N` | `R` (upper bound on `C1`) |
|---|---|---|
| Flat indicator (this scaffold's ceiling) | any | 2.0 |
| Previous published record (pre-2025) | — | 1.5098 |
| AlphaEvolve (App. B.1) | 600 | 1.5053 |
| TTT-Discover | 30000 | 1.5028628983 |
| **AutoEvolver record** (Claude/Opus) | 30000 | **1.5028628969** |
| Provable lower bound (Cloninger–Steinerberger) | — | 1.28 (floor, not an `f`) |

The record was reached by dedicated search spending enormous compute (AutoEvolver ran ~40 h of autonomous
optimization; the gain from `1.5053` to `1.50286` is in the *fourth* decimal place). Reproducing `1.5028628969`
from scratch in a single bounded constructor is not expected — so the ladder here is not chasing a known-easy target,
it is descending from the trivial flat ceiling at `2.0` toward the published frontier, and the gap from wherever a
single local constructor lands down to `1.50286` (and the further gap to the `1.28` floor) is the honest measure of
how open the problem still is.

## The related constants (for orientation)

This task focuses on `C1` because it is cleanly runnable, but it sits in a small family. The **second**
autocorrelation inequality (`C2`) is the *maximization* sibling — the largest `R = ||f*f||_2^2 / (||f*f||_inf ·
||f*f||_1)`, where *higher* is better; its record is `C2 ≥ 0.96102` (AlphaEvolve-V2). The **third** (`C3`) is a
related ratio variant. Only `C1` is built here; `C2`/`C3` are cited as the surrounding landscape. The two problems
share the same step-function substrate and the same FFT-based evaluator, but they pull in opposite directions: `C2`
flattens the autoconvolution's cap to make it indicator-like, while `C1` pushes the peak *down* relative to the mass.

## The fixed substrate

The harness is a thin, deterministic evaluator. It receives a height vector `a` (a Python list of non-negative
floats), clips negatives to `0` and values to `1000`, forms the autoconvolution `b = a*a` by self-convolution,
takes the peak `max_k b_k`, and returns `R = 2N · max_k b_k / (Σ a_n)^2`. Self-convolution is done with an FFT
(`scipy.signal.fftconvolve`), so the evaluator is `O(N log N)` and scales to tens of thousands of pieces — the same
speedup AutoEvolver used to reach `N = 30000`. The formula, the clipping, and `R ≥ C1 ≥ 1.28` are frozen.

```python
import numpy as np
from scipy.signal import fftconvolve

def autoconv_c1_ratio(a):
    """R = 2N * max(a*a) / (sum a)^2 for f = sum_n a_n 1_[n,n+1). Lower = tighter upper bound on C1."""
    a = np.clip(np.asarray(a, dtype=float), 0.0, 1000.0)
    N = len(a)
    b = fftconvolve(a, a)                  # length 2N-1: b_k = sum_n a_n a_{k-n}
    peak = float(np.max(b))                # max of the piecewise-linear autoconvolution = max node
    s = float(np.sum(a))
    if s < 0.01:
        return float("inf")
    return 2.0 * N * peak / (s * s)
```

Every valid candidate is a list of non-negative finite floats of any length `N ≥ 1`. There are no other
constraints — the constructor is free to return any height vector, structured, optimized, or searched.

## Evaluation settings

A single deterministic functional `R(a)` above. Because a constructor may search internally, the harness reports
`R` of the *returned* vector and fixes any stochastic run to a stated seed so the number is reproducible. The fixed
yardsticks — flat ceiling `2.0`, AlphaEvolve `1.5053`, TTT-Discover `1.5028628983`, AutoEvolver record
`1.5028628969`, provable floor `1.28` — are the rulers every rung is read against. The score is the whole result;
there is no partial credit beyond the ratio itself, and **lower is better**.
