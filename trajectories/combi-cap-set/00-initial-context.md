## Research question

A **cap set** in `F_3^n` is a subset of `{0,1,2}^n` containing no three distinct points `a, b, c` with `a + b + c ≡ 0 (mod 3)`. Equivalently, it contains no three-term arithmetic progression and no three collinear points. The task is to build a **constructor** — a program that emits one concrete cap set — and score it by the size of the valid cap it returns.

The best known sizes are:

| `n` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| best known `|cap|` | 2 | 4 | 9 | 20 | 45 | 112 | 236 | ≥ 496 |

The values through `n = 7` are proven optima. At `n = 8` the best known construction has size `496`; the exact maximum is open. The polynomial-method cap-set theorem gives the upper bound `|cap| ≤ O(2.756^n)`, so caps are exponentially sparse in `3^n`. The challenge is to push the lower bound at each finite `n`.

## Prior art / Background / Baselines

- **Affine and product constructions.** Small caps lift to larger ones by taking products and unions across coordinates, giving many of the classical lower bounds. *Gap:* they require a separate algebraic construction for each `n` and stall far below the upper-bound constant; the exact optima at `n ≤ 7` reflect exhaustive or dedicated computer search, not a closed form.
- **Croot–Lev–Pach / Ellenberg–Gijswijt cap-set theorem.** The polynomial method proves `|cap| ≤ O(2.756^n)`. *Gap:* it bounds how large a cap can be but constructs nothing; matching lower-bound constructions at finite `n` remain the open, search-driven side.
- **Greedy and priority-based search.** Order the `3^n` vectors by some priority and add each one greedily if it does not close a line; the order can be fixed, random, or symmetric. *Gap:* hand-designed priorities plateau below the known optima, and finding an order that reaches the strong constructions is the central difficulty.

## Fixed substrate / Code framework

The harness is a deterministic evaluator. It takes a constructor's output, verifies that it is a valid cap, and reports its size. The verifier below uses an incremental `O(|cap|^2 · n)` block check: each point is assigned an integer index, and as points are admitted, the unique third point that would complete a line with any earlier point is marked blocked. For small `n` the harness also runs an independent `O(|cap|^3)` triple scan.

```python
import itertools
import numpy as np

def all_vectors(n):
    return np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)

def is_cap_set(vectors):
    """Valid cap iff no three distinct points sum to 0 mod 3. O(c^2 n) incremental check."""
    if len(vectors) == 0:
        return True
    vectors = np.asarray(vectors, dtype=np.int32)
    c, n = vectors.shape
    powers = np.array([3 ** j for j in range(n - 1, -1, -1)], dtype=np.int64)
    raveled = vectors.astype(np.int64) @ powers
    if len(set(raveled.tolist())) != c:
        return False                                   # no duplicate points
    is_blocked = np.full(3 ** n, False, dtype=bool)
    for i, (v, idx) in enumerate(zip(vectors, raveled)):
        if is_blocked[idx]:
            return False                               # adding v would close a line
        if i >= 1:
            blocking = ((-vectors[:i, :] - v[None, :]) % 3).astype(np.int64) @ powers
            is_blocked[blocking] = True
        is_blocked[idx] = True
    return True
```

## Editable interface

The editable part is the **constructor**: a Python function that takes `n` and returns a list of distinct vectors in `{0,1,2}^n`. The harness calls the constructor, runs `is_cap_set` on the returned list, and scores it by `|cap|` if it is valid. The constructor is free to use any strategy — greedy ordering, random restart, structured priority, algebraic lifting, or something else — as long as the final set passes the verifier.

## Evaluation settings

A single deterministic objective per `n`: the size `|cap|` of the returned valid cap. If the constructor is stochastic, the harness fixes the random seed so the result is reproducible. An invalid set scores nothing. The yardsticks are the trivial `2^n` floor, the proven optima `20 / 45 / 112 / 236` at `n = 4..7`, and the current best construction `496` at `n = 8`.
