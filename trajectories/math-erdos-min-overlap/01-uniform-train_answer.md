I want a step function on $[0,2]$, given as $n$ equal-width cells with heights $v_i \in [0,1]$ summing to $n/2$, whose worst overlap against its own complement $C(v) = \max_k \big(\sum_i v_i(1-v_{i-k})\big)\cdot 2/n$ is as small as possible — lower is a tighter upper bound on the Erdős minimum-overlap constant $C5$, which the literature pins into the sliver $0.379005 \le C5 \le 0.380869$. Before I try to be clever I need to fix the floor: a baseline I understand by hand, that every later rung must beat, and that tells me what a value of this functional even feels like.

The method here is the *flat half-density profile*: every cell exactly half-full, $v_i \equiv 1/2$. It is the only profile in the class with no internal structure at all, and its overlap I can compute entirely by hand. The complement of a flat profile is itself — if $v_i = 1/2$ everywhere then $1 - v_i = 1/2$ everywhere too — so the overlap at a shift $k$ is a sum of $(1/2)(1/2) = 1/4$ over however many cells line up at that shift. The number of overlapping cells is largest at zero shift, where all $n$ cells coincide, so the worst overlap is

$$\max_k c_k = c_0 = \sum_{i} \tfrac14 = \tfrac{n}{4},\qquad C = \tfrac{n}{4}\cdot\tfrac{2}{n} = \tfrac12.$$

This is exactly Erdős's own 1955 upper bound $C5 \le 1/2$, and what makes it the right thing to start from is that the value is *forced*, not chosen. The balance constraint $\sum_i v_i = n/2$ pins the average height to $1/2$, and among all feasible profiles the flat one is the unique point with zero internal variation — it sits dead center of the feasible region (its sum is $n\cdot\tfrac12 = n/2$, exactly on the constraint), so the floor I measure is a genuine interior point, not a degenerate corner. Crucially it is *invariant to the piece count*: a flat vector of $10$ half-cells and one of $1000$ half-cells both have the same triangular overlap envelope peaking at $n/4$ and both score $1/2$ exactly. Refining a flat profile buys nothing, because every refinement of $1/2$-cells is still all $1/2$-cells.

That invariance is the real lesson, and it explains why the problem is hard. The piece count $n$ is not itself a lever; only the *shape* of the heights moves the bound. To get the worst $c_k$ below $1/2$ I have to break the perfect self-alignment at zero shift — push some cells toward $0$ and others toward $1$, asymmetrically, so that where $A$-mass is heavy $B$-mass is light and the products $v_i(1-v_{i-k})$ cannot all be large at once (a cell at $1$ contributes nothing when aligned with another cell at $1$, since $1-1=0$; a cell at $0$ contributes nothing as the $v_i$ factor). The constraint forbids the trivial escape of shrinking every height — the mass has to go somewhere — so the only move is *redistribution* into a near-binary, spiky profile. The flat function has nothing to vary; it is a rigid baseline parked at the top of the achievable range with no degree of freedom to spend. The whole distance from $0.5$ down to $\sim 0.3809$ is what later searched rungs must buy, and the cleanest place to begin that search is at a small piece count where a constrained optimizer can explore the shape space thoroughly.

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
