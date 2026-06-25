**Problem.** Construct a step function `h` on `[0,2]`, given as `n` equal-width cells with heights `v_i ∈
[0,1]` summing to `n/2`, minimizing the worst overlap `C(v) = max_k (Σ_i v_i (1 − v_{i−k})) · 2/n` — an
upper bound on the Erdős minimum-overlap constant `C5` (LOWER is better; the true `C5 ∈ [0.379005,
0.380869]`). The starting fill is the most symmetric feasible profile.

**Key idea.** Take the flat half-density `v_i ≡ 1/2`: every cell exactly half-full, the density of `A`
equal to the density of `B` everywhere. Its complement is itself, so at every shift each overlapping cell
contributes `(1/2)(1/2) = 1/4`; the worst shift is `k = 0`, where all `n` cells line up, giving overlap
`n/4` and bound `(n/4)·2/n = 1/2`. This is exactly Erdős's own 1955 upper bound `C5 ≤ 1/2`.

**Why these choices.** The construction is forced, not chosen: the balance constraint `Σ v = n/2` pins
the average height to `1/2`, and the flat profile is the unique feasible point with no internal
variation. It is deliberately the *floor* — a rigid, parameter-free baseline guaranteed feasible (sum is
exactly `n/2`) that hands back the historically-first bound. It is invariant to the piece count `n`:
refining a flat profile into more half-cells changes nothing, because the overlap envelope is the same
triangle peaking at `n/4`. It is not expected to approach the record — every modern bound comes from
*breaking* this symmetry into a near-binary spiky profile — so the entire distance from `0.5` down to
`~0.3809` is what later, searched rungs must buy.

**Hyperparameters / contract.** One knob `n` (piece count), default `n = 600` to match the resolution of
the published records; the value is `0.5` for every `n`. Output is a feasible height vector (all `1/2`,
sum exactly `n/2`). Deterministic — same vector every call.

```python
import numpy as np

def construct(n=600):
    """Erdos minimum-overlap, rung 1: flat half-density floor.
    Returns n cells each = 1/2 (feasible: sum == n/2). Evaluates to C = 0.5 for every n
    (Erdos's 1955 upper bound). Lower C is a tighter bound on the constant C5."""
    return np.full(n, 0.5)

# --- frozen evaluator ---
def compute_upper_bound(sequence):
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')   # c_k = sum_i v_i (1 - v_{i-k})
    return float(np.max(conv) / len(seq) * 2.0)

if __name__ == "__main__":
    v = construct()
    assert abs(v.sum() - len(v) / 2.0) < 1e-9          # balance constraint
    print("n =", len(v), " C =", compute_upper_bound(v))   # -> C = 0.5
```
