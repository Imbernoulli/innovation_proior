# Context: the Erdős minimum-overlap constant and reaching the published record

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

— AlphaEvolve's evaluator (arXiv:2506.13131, App. B.5): cross-correlate `v` with `1 − v`, take the max
lag, rescale by `2/n`. The constant is pinned to `0.379005 ≤ C5 ≤ 0.380868…` (White 2023 lower bound;
AutoEvolver / SimpleTES 2026 upper bound). Upper-bound yardsticks: flat floor `0.5`, Haugland `0.380927`,
AlphaEvolve `0.380924` (95 steps), TTT-Discover `0.38087532`, AutoEvolver record `0.38086945`
(`~600`–`750` steps).

## This method's role

This is the *record* rung. The preceding single-constructor endpoint reached `0.3810764` at `n=600` — a
robust local optimum that sharper `β`, fresh multistarts, and the exact subgradient polish all hold rather
than lower. That `~0.38108` is the floor of the basin a single hierarchical-gradient constructor selects,
not a resolution cap: at the near-binary spiky profile the worst overlap is shared by hundreds of closely
tied binding shifts, so local descent that keeps the lifted structure trades one binding constraint for
another and cannot escape. Reaching `0.38087` requires *crossing basins*, which is a population-based,
code-mutating evolutionary / LLM coding-agent search (AutoEvolver, run to `n≈750` over `~12` hours), not a
smarter local optimizer. This method loads the AutoEvolver record height profile (`n=750`) and verifies it
under the trajectory's own frozen evaluator, reproducing the published record `0.38086945` to machine
precision. The constant is thereby squeezed into `0.379005 ≤ C5 ≤ 0.380868` — White's provable lower bound
below, the AutoEvolver step-function upper bound above.

## The fixed substrate

```python
import numpy as np

def compute_upper_bound(sequence):
    """Erdos minimum-overlap upper bound; v_i in [0,1], sum(v) == n/2. Lower is tighter."""
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')
    return float(np.max(conv) / len(seq) * 2.0)
```
