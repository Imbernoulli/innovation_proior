# Context: the Erdős minimum-overlap constant and its step-function formulation

## Research question

Erdős (1955) asked: split `{1, …, 2n}` into two equal halves `A` and `B`; for an integer shift `k`,
count the pairs `(a, b)` with `a ∈ A`, `b ∈ B`, `a − b = k`; let `M(n)` be the largest such count over
`k`, minimized over all balanced splits. Erdős proved `M(n)/n` tends to a constant `C5`, and the value of
that constant is the open question. Haugland (2016) proved `C5` equals the infimum, over all step
functions `h` on `[0,2]` with values in `[0,1]` and `∫_0^2 h = 1`, of `max_k ∫ h(x)(1 − h(x+k)) dx` —
the worst overlap of the `A`-density `h` against the `B`-density `1 − h`. Every explicit step function
gives an *upper bound* on `C5`; LOWER is better.

## How the score is defined

A candidate is a length-`n` vector `v` of cell heights in `[0,1]` with `Σ v = n/2` (the balance
constraint, half the mass on each side). The discrete worst overlap is

```
C(v) = ( max_k Σ_i v_i (1 − v_{i−k}) ) · 2 / n
```

This is AlphaEvolve's published evaluator (arXiv:2506.13131, App. B.5): cross-correlate `v` with `1 − v`,
take the max lag, rescale by `2/n`. The true constant is pinned to `0.379005 ≤ C5 ≤ 0.380868…` (White's
2023 convex-programming lower bound; SimpleTES 2026 upper bound), a window of width `~2×10^{-3}`.
Yardsticks: Haugland `0.380927` (2016), AlphaEvolve `0.380924`, TTT-Discover `0.38087532`, AutoEvolver
record `0.38086945`.

## This method's role

This is the *floor* rung: the flat half-density `v_i ≡ 1/2`, the most symmetric feasible profile, whose
worst overlap is exactly `1/2` — Erdős's own 1955 upper bound. It establishes the starting altitude that
every optimized rung must beat, and exposes that the piece count alone is no lever: only the *shape* of
the heights moves the bound.

## The fixed substrate

```python
import numpy as np

def compute_upper_bound(sequence):
    """Erdos minimum-overlap upper bound; v_i in [0,1], sum(v) == n/2. Lower is tighter."""
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')
    return float(np.max(conv) / len(seq) * 2.0)
```
