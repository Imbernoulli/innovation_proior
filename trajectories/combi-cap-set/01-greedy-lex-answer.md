**Problem.** Construct a large cap set in `F_3^n` — a set of vectors in `{0,1,2}^n` with no three
distinct points summing to `0 (mod 3)` (no line / no 3-term AP). The constructor returns the set;
it is scored by `|cap|` only if the set is a *valid* cap, verified by the harness. This rung is the
floor: the simplest deterministic rule that is guaranteed to emit a valid cap.

**Key idea.** Walk the `3^n` vectors in fixed **lexicographic** order and greedily admit each one
if and only if it does not complete a line with two already-admitted points. Over `F_3` every pair
of points `a, b` determines a unique third point `r = (−a − b) mod 3` that would close the line
`{a, b, r}`; so admitting a point and then blocking the closing-point of every line it forms with
an existing point keeps the running set a cap by construction. The result is correct for any walk
order; lexicographic is the parameter-free default.

**Why these choices.** Validity is free here — every admitted point was checked against all
predecessors at admission, so the output is a cap no matter what order is used. The only degree of
freedom is the *order*, and lexicographic is deliberately the most naive one: deterministic, no
seed, no tuning. It is geometry-blind — the counting order is an artifact of how tuples are
enumerated, not aligned with the structure of `F_3^n` — so it commits early and locally and cannot
look ahead. That is exactly why it is the right floor: it pins down the trivial `2^n`-type baseline
that every smarter ordering must beat, and it isolates "the ordering" as the single lever the rest
of the ladder will pull.

**Hyperparameters / contract.** None. Deterministic — same cap every call. Output is a list of
distinct vectors in `{0,1,2}^n` satisfying the cap property; the incremental block set guarantees
validity. Cost is `O(|cap|^2 · n)` to build.

```python
import itertools
import numpy as np

def construct(n):
    """Greedy lexicographic cap set in F_3^n. Add each vector in counting order iff it
    keeps the set cap-valid. Returns an (m, n) int array of distinct vectors."""
    av = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)
    powers = np.array([3 ** j for j in range(n - 1, -1, -1)], dtype=np.int64)
    raveled = av.astype(np.int64) @ powers
    is_blocked = np.full(3 ** n, False, dtype=bool)   # third point of each closed line
    cap = []
    capset = np.empty((0, n), dtype=np.int32)
    for v, idx in zip(av, raveled):
        if is_blocked[idx]:
            continue                                  # would complete a forbidden line
        if len(cap) >= 1:
            # block the closing point r = (-q - v) mod 3 of every line {q, v, r}
            blocking = ((-capset - v[None, :]) % 3).astype(np.int64) @ powers
            is_blocked[blocking] = True
        is_blocked[idx] = True
        cap.append(v)
        capset = np.concatenate([capset, v[None, :]], axis=0)
    return np.array(cap, dtype=np.int32)
```
