## Research question

Split `{1, 2, …, 2n}` into two equal halves `A` and `B`. For an integer shift `k`, count the pairs `(a, b)` with `a ∈ A`, `b ∈ B` and `a − b = k`; let `M(n)` be the largest such count over all `k`, minimized over all balanced splits. Erdős (1955) proves `M(n)/n` tends to a constant and asks for that constant's value. Equivalently, the **Erdős minimum-overlap constant** `C5` is the largest constant such that

```
sup_{x ∈ [-2,2]}  ∫_{-1}^{1} f(t) · g(x+t) dt  ≥  C5
```

for every pair of non-negative functions `f, g : [-1,1] → [0,1]` with `f + g = 1` on `[-1,1]` and `∫ f = 1` (extended by zero outside `[-1,1]`). Here `f` is the density of `A` and `g = 1 − f` the density of `B`; the convolution is the overlap of the `A`-mass with the shifted `B`-mass, and `sup_x` is the worst overlap. Every feasible `f` certifies `C5 ≤ (that overlap)`; lower is better.

Haugland (2016) shows the continuous constant equals the infimum, over all step functions `h` on `[0, 2]` with values in `[0, 1]` and `∫_0^2 h = 1`, of `max_k ∫ h(x)(1 − h(x+k)) dx`. So a candidate is a length-`n` height vector `v = (v_0, …, v_{n-1})` with

```
Σ_i v_i = n/2
```

and the overlap at lag `k` is the discrete cross-correlation

```
c_k = Σ_i v_i · (1 − v_{i−k})
```

The score is the worst overlap, normalized to the continuous limit:

```
C(v) = ( max_k c_k ) · 2 / n
```

`C(v)` is a deterministic functional of the heights; any feasible `v` gives an upper bound on `C5`.

## Prior art / Background / Baselines

| Reference point | `C` value | Role |
|---|---|---|
| White lower bound (2023) | 0.379005 | proven floor on `C5` |
| Haugland (2016) | 0.380927 | best published step-function upper bound |
| Motzkin–Ralston–Selfridge (1956) | 0.4 | early hand-optimized step function |
| Erdős flat bound (1955) | 0.5 | trivial uniform-density upper bound |

- **Erdős flat bound (1955)**: set every height to `1/2`. This is the simplest balanced split; it gives `C = 0.5`.
- **Motzkin–Ralston–Selfridge (1956)**: a hand-optimized step-function pattern that improves the bound to `0.4`.
- **Haugland (1993, 1996, 2016)**: a sequence of increasingly refined step functions, culminating in `C ≈ 0.380927`.
- **White (2023)**: proves `C5 ≥ 0.379005` by convex programming.

## Fixed substrate / Code framework

The harness is a thin, deterministic evaluator. It receives a height vector `v`, checks the balance constraint `Σ v = n/2`, forms the cross-correlation of `v` with `1 − v`, takes the maximum lag, and rescales by `2/n`.

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

## Editable interface

The constructor returns a single feasible height vector: a Python list (or 1-D array) of floats in `[0, 1]` of any length `n ≥ 1`, with `Σ_i v_i = n/2`. There are no other constraints; the vector may be structured, optimized, searched, or randomized, as long as it passes `verify_sequence`.

## Evaluation settings

The result is the single number `C(v)` returned by `compute_upper_bound`. If a constructor uses randomness, the run is fixed to a stated seed so the score is reproducible. The yardsticks are the baselines above: Erdős flat `0.5`, Motzkin–Ralston–Selfridge `0.4`, Haugland `0.380927`, and White `0.379005`. There is no partial credit beyond the worst-overlap value itself.
