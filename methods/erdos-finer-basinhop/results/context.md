# Context: the Erdős minimum-overlap constant and its step-function formulation

## Research question

Erdős (1955) asked: split `{1, …, 2n}` into two equal halves `A` and `B`; for an integer shift `k`,
count the pairs `(a, b)` with `a ∈ A`, `b ∈ B`, `a − b = k`; let `M(n)` be the largest such count over
`k`, minimized over all balanced splits. Erdős proved `M(n)/n → C5`. Haugland (2016) proved `C5` equals
the infimum, over all step functions `h` on `[0,2]` with values in `[0,1]` and `∫_0^2 h = 1`, of `max_k
∫ h(x)(1 − h(x+k)) dx` — the worst overlap of the `A`-density `h` against the `B`-density `1 − h`. Every
explicit step function is an *upper bound* on `C5`; LOWER is better.

## How the score is defined

A candidate is a length-`n` vector `v` of cell heights in `[0,1]` with `Σ v = n/2`. The discrete worst
overlap is

```
C(v) = ( max_k Σ_i v_i (1 − v_{i−k}) ) · 2 / n
```

— AlphaEvolve's evaluator (arXiv:2506.13131, App. B.5). The constant is pinned to `0.379005 ≤ C5 ≤
0.380868…`. Yardsticks: flat floor `0.5`, Haugland `0.380927`, AlphaEvolve `0.380924`, TTT-Discover
`0.38087532`, AutoEvolver record `0.38086945`.

## This method's role

This is the *middle-resolution* rung. The coarse SLSQP rung reached `~0.3812` at `24` cells but is capped
by resolution. This method lifts the optimized coarse profile to `~120` cells by upscaling (a free
no-op that preserves `C` while adding degrees of freedom), then refines with **basin-hopping** around the
annealed-SLSQP ladder — perturb-and-re-solve, accepting only improvements — with a sharper `β` schedule
to track the spikier overlap envelope. It confirms that lifting helps and tunes the hop schedule before
the endpoint rung lifts to the `~600`-cell scale of the published records.

## The fixed substrate

```python
import numpy as np

def compute_upper_bound(sequence):
    """Erdos minimum-overlap upper bound; v_i in [0,1], sum(v) == n/2. Lower is tighter."""
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')
    return float(np.max(conv) / len(seq) * 2.0)
```
