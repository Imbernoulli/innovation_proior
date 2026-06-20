## Research question

Split `{1, 2, …, 2n}` into two equal halves `A` and `B`. For an integer shift `k`, count the pairs
`(a, b)` with `a ∈ A`, `b ∈ B` and `a − b = k`; let `M(n)` be the *largest* such count over all `k`,
minimized over all balanced splits. Erdős (1955) asked for the growth rate: he proved `M(n)/n` tends to
a constant, and the question is the value of that constant. Equivalently, in the continuous limit, the
**Erdős minimum-overlap constant** `C5` is the largest constant such that

```
sup_{x ∈ [-2,2]}  ∫_{-1}^{1} f(t) · g(x+t) dt  ≥  C5
```

for *every* pair of non-negative functions `f, g : [-1,1] → [0,1]` with `f + g = 1` on `[-1,1]` and
`∫ f = 1` (extended by zero outside `[-1,1]`). Here `f` is the density of `A` and `g = 1 − f` the
density of `B`; the convolution `∫ f(t) g(x+t) dt` is exactly the overlap of the `A`-mass with the
shifted `B`-mass, and `sup_x` of it is the worst (largest) overlap. Every construction one writes down
is therefore an **upper bound** on `C5` — pick a concrete `f`, measure the worst overlap, and that
number certifies `C5 ≤ (that number)`. LOWER is better. The single object being designed is exactly
that `f`, scored by the worst-overlap functional alone.

The construction class is the one used by every modern result on this problem. Haugland (2016) proved
the continuous constant equals the infimum, over all **step functions** `h` on `[0, 2]` with values in
`[0, 1]` and `∫_0^2 h = 1`, of `max_k ∫ h(x)(1 − h(x+k)) dx`. So a candidate is a non-negative
piecewise-constant profile — `n` equal-width pieces with heights in `[0,1]` — and the score is the worst
overlap of `h` against its own complement `1 − h`. This is what makes the problem cleanly computable:
the overlap is a discrete cross-correlation of the height vector with its complement, so it is fully
determined by the heights at the `n` cells.

## How the score is defined

Write the candidate as a length-`n` vector `v = (v_0, …, v_{n-1})` of heights in `[0, 1]`. The
balanced-split constraint `∫ h = 1` (half the total mass on each side) becomes, on `n` equal cells,

```
Σ_i v_i = n / 2     (the sum is exactly n/2, exactly half the cells' worth of mass)
```

The overlap of `h` with its shifted complement, at integer cell-shift `k`, is the discrete
cross-correlation

```
c_k = Σ_i v_i · (1 − v_{i−k})     (sum over the valid overlap range)
```

and the worst overlap, normalized to the continuous limit, is

```
C(v) = ( max_k c_k ) · 2 / n
```

This is the literal evaluator. `max_k c_k` is the worst alignment of the `A`-mass against the
complement `B`-mass; the factor `2/n` rescales the discrete count to the `[0,2]`-interval continuous
constant. There is no held-out set and no way to game the metric: `C(v)` is a deterministic functional
of the heights, and any feasible `v` certifies `C5 ≤ C(v)`. Lower `C(v)` is a tighter (better) bound.

A few fixed yardsticks anchor every rung. The **flat** profile `v_i ≡ 1/2` (every cell half-full)
gives `c_k = Σ_i (1/2)(1/2) = n/4` at the center, so `C = (n/4)·2/n = 1/2` — this is Erdős's own 1955
upper bound `0.5`, and it is the floor this scaffold starts from. Decades of hand-optimized step
functions brought the bound down: Motzkin–Ralston–Selfridge `0.4` (1956), Haugland `0.385694` (1993),
`0.382002` (1996), and `0.380927` (2016). Google DeepMind's **AlphaEvolve** found an alternative step
function giving `0.380924` (arXiv:2506.13131, App. B.5). **TTT-Discover** reached `0.38087532`
(arXiv:2601.16175) and **AutoEvolver** `0.38086945` (https://tengxiaoliu.github.io/autoevolver/),
both beating AlphaEvolve; the very latest reported values sit near `0.380868` (SimpleTES, 2026). On the
*lower* side, White (2023) proved `C5 ≥ 0.379005` by convex programming, so the true constant is pinned
to `0.379005 ≤ C5 ≤ 0.380868…` — a window of width `~2×10^{-3}`. The headline numbers to keep in view:

| Reference point | pieces `n` | `C` (upper bound) |
|---|---|---|
| White lower bound (provable floor on `C5`) | — | 0.379005 |
| **AutoEvolver record** (step fn) | ~600–750 | **0.38086945** |
| TTT-Discover | — | 0.38087532 |
| AlphaEvolve | 95 | 0.380924 |
| Haugland (2016) | — | 0.380927 |
| Erdős flat-density bound (this scaffold's floor) | any | 0.5 |

The record has been ground down by dedicated search spending enormous compute (AutoEvolver ran to
`n=750` over `~12` hours of evolutionary coding-agent search; AlphaEvolve is an evolutionary search).
Reproducing `0.380869` from scratch in a single bounded constructor is not expected — so the ladder here
is not chasing a known-easy target, it is climbing from Erdős's flat floor toward the published frontier,
and the gap between what a single optimizer reaches and `0.380869` is the honest measure of how tight the
problem already is. The constant is squeezed into a window of width `~2×10^{-3}`; every digit past the
third costs real work.

## The fixed substrate

The harness is a thin, deterministic evaluator — it is AlphaEvolve's own published verifier
(arXiv:2506.13131, App. B.5). It receives a height vector `v` (a Python list of floats in `[0,1]`),
checks the balance constraint `Σ v = n/2`, forms the cross-correlation of `v` with `1 − v`, takes the
maximum lag, and rescales by `2/n`. The cross-correlation is `O(n^2)` by direct `np.correlate` (or
`O(n log n)` by FFT for large `n`); both give the same number. The formula, the constraint, and the
`2/n` normalization are frozen.

```python
import numpy as np

def compute_upper_bound(sequence):
    """Erdos minimum-overlap upper bound for h = sum_i v_i 1_[cell i],
    v_i in [0,1], sum(v) == n/2. Lower is a tighter bound. (AlphaEvolve App. B.5.)"""
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')   # c_k = sum_i v_i (1 - v_{i-k})
    return float(np.max(conv) / len(seq) * 2.0)         # worst overlap, rescaled to [0,2]

def verify_sequence(sequence, tol=1e-6):
    """Feasibility: values in [0,1], sum exactly n/2."""
    seq = np.asarray(sequence, dtype=float)
    assert np.all(seq >= -1e-9) and np.all(seq <= 1 + 1e-9)
    assert abs(seq.sum() - len(seq) / 2.0) < tol * max(1, len(seq))
    return True
```

Every valid candidate is a list of floats in `[0,1]` of any length `n ≥ 1` summing to `n/2`. There are
no other constraints — the constructor is free to return any feasible height vector, structured,
optimized, or searched.

## Evaluation settings

A single deterministic functional `C(v)` above. Because a constructor may search internally, the harness
reports `C` of the *returned* vector and fixes any stochastic run to a stated seed so the number is
reproducible. The fixed yardsticks — Erdős flat floor `0.5`, Haugland `0.380927`, AlphaEvolve
`0.380924`, TTT-Discover `0.38087532`, AutoEvolver record `0.38086945`, White lower bound `0.379005` —
are the rulers every rung is read against. The score is the whole result; there is no partial credit
beyond the worst-overlap number itself.
