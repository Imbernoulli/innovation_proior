# Context: the Erdős minimum-overlap constant and the published record

## Research question

Erdős (1955) asked: split `{1, …, 2n}` into two equal halves `A` and `B`; for an integer shift `k`,
count the pairs `(a, b)` with `a ∈ A`, `b ∈ B`, `a − b = k`; let `M(n)` be the largest such count over
`k`, minimized over all balanced splits. Erdős proved `M(n)/n → C5`. Haugland (2016) proved `C5` equals
the infimum, over all step functions `h` on `[0,2]` with values in `[0,1]` and `∫_0^2 h = 1`, of `max_k
∫ h(x)(1 − h(x+k)) dx` — the worst overlap of the `A`-density `h` against the `B`-density `1 − h`. Every
explicit step function is an *upper bound* on `C5`; LOWER is better.

The setting here: given the discrete evaluator below, find a feasible height profile whose worst overlap
is as small as possible, pushing the upper bound on `C5` as low as the current record.

## How the score is defined

A candidate is a length-`n` vector `v` of cell heights in `[0,1]` with `Σ v = n/2`. The discrete worst
overlap is

```
C(v) = ( max_k Σ_i v_i (1 − v_{i−k}) ) · 2 / n
```

— AlphaEvolve's evaluator (arXiv:2506.13131, App. B.5): cross-correlate `v` with `1 − v`, take the max
lag, rescale by `2/n`. The constant is pinned to `0.379005 ≤ C5 ≤ 0.380868…` (White 2023 lower bound;
AutoEvolver / SimpleTES 2026 upper bound). Upper-bound yardsticks: flat floor `0.5`, Haugland `0.380927`,
AlphaEvolve `0.380924` (95 steps), TTT-Discover `0.38087532`, AutoEvolver record `0.38086945`
(`~600`–`750` steps).

## Where the trajectory stands

The preceding single-constructor endpoint on this ladder reached `0.3810764` at `n=600` with a
hierarchical-gradient constructor: at that point the profile is near-binary and spiky, with the worst
overlap shared by hundreds of closely tied binding shifts. The published record stands at `0.38086945`,
obtained by AutoEvolver run to `n≈750` over `~12` hours. The goal here is to reach that published record
number `0.38086945` under this trajectory's own frozen evaluator, squeezing `C5` into
`0.379005 ≤ C5 ≤ 0.380868` — White's provable lower bound below, the step-function upper bound above.

## The fixed substrate

```python
import numpy as np

def compute_upper_bound(sequence):
    """Erdos minimum-overlap upper bound; v_i in [0,1], sum(v) == n/2. Lower is tighter."""
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')
    return float(np.max(conv) / len(seq) * 2.0)
```
