The Erdős minimum-overlap problem asks for the asymptotic constant `C5` governing how unavoidably two
equal halves of `{1, …, 2n}` must overlap under shifts, and Haugland's theorem lets me work entirely with
its clean continuous proxy: a step function `h` on `[0,2]` with values in `[0,1]` and integral `1`, scored
by its worst overlap `max_k ∫ h(x)(1 − h(x+k)) dx` against its own complement `1 − h`. Discretized into
`n` equal cells with heights `v_i ∈ [0,1]` summing to `n/2`, the score is the largest lag of the
cross-correlation of `v` with `1 − v`, rescaled by `2/n`. Lower is a tighter upper bound on `C5`. Before
optimizing anything, I want the floor — the simplest legal profile — so I know what a value of this
functional means and so every later rung has a baseline to beat.

The method here is the flat half-density: set every cell to `1/2`. It is the only feasible profile with no
internal structure at all, and it sits exactly on the balance constraint, since `n` cells of height `1/2`
sum to `n/2`. Its worst overlap I can compute by hand. The complement of the flat profile is itself, so at
every shift each overlapping cell contributes `(1/2)(1/2) = 1/4`; the worst shift is zero shift, where all
`n` cells line up, giving overlap `n/4`. Rescaled by `2/n` this is exactly `1/2`. That is Erdős's own 1955
upper bound `C5 ≤ 0.5`, recovered as a pure closed form. Crucially the value is independent of the piece
count: a flat vector of ten half-cells and one of a thousand half-cells both have the same triangular
overlap envelope peaking at `n/4`, so both score `1/2`. The number of pieces is a red herring on its own —
only the *shape* of the heights moves the bound.

This is deliberately a rigid, parameter-free floor. The balance constraint pins the average height to
`1/2`, and the flat profile is the unique feasible point with zero variation, so there is nothing to tune
and no direction to descend: the worst overlap is locked to `1/2` by the perfect self-alignment of `A`-mass
with `B`-mass at zero shift. Every modern bound — Haugland's `0.380927`, AlphaEvolve's `0.380924`,
AutoEvolver's record `0.38086945`, all the way down toward White's provable lower limit `0.379005` — comes
from *breaking* this symmetry into an asymmetric, near-binary profile that makes the products
`v_i(1 − v_{i−k})` small at every shift simultaneously. The entire distance from `0.5` down to `~0.3809` is
what the later searched rungs must buy; this rung just establishes the starting altitude and confirms the
evaluator agrees with the hand computation.

```python
import numpy as np

def construct(n=600):
    """Erdos minimum-overlap, floor rung: flat half-density.
    Returns n cells each = 1/2 (feasible: sum == n/2). Evaluates to C = 0.5 for every n
    (Erdos's 1955 upper bound). Lower C is a tighter bound on the constant C5."""
    return np.full(n, 0.5)

def compute_upper_bound(sequence):
    """Frozen evaluator (AlphaEvolve, App. B.5)."""
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')   # c_k = sum_i v_i (1 - v_{i-k})
    return float(np.max(conv) / len(seq) * 2.0)

if __name__ == "__main__":
    v = construct()
    assert abs(v.sum() - len(v) / 2.0) < 1e-9           # balance constraint
    print("n =", len(v), " C =", compute_upper_bound(v))   # -> C = 0.5
```
