# Context: the cap-set problem and greedy construction

## Research question

A **cap set** in `F_3^n` (length-`n` vectors over the integers mod 3) is a subset with no
line: no three *distinct* points `a, b, c` with `a + b + c ≡ 0 (mod 3)` — equivalently, no
three-term arithmetic progression and no three collinear points (over `F_3` the line through
`a, b` is `{a, b, −a−b}`, so every pair has exactly one "completing" third point). The
question is the central one in the subject: **how large can a cap set in `F_3^n` be?** The task
here is to design a *constructor* — a program emitting one concrete subset — scored by the cap
size it returns, but only if the returned set is a *verified* valid cap.

The maximal sizes are known exactly only for small `n` and grow near `2.756^n` (the
Croot–Lev–Pach / Ellenberg–Gijswijt upper bound is `O(2.756^n)`):

| `n` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| max `|cap|` | 2 | 4 | 9 | 20 | 45 | 112 | 236 | ≥ 512 |

Values through `n = 6` are proven optima; `n = 7` is `236`; at `n = 8` the best known lower
bound is `512`.

## How the score is defined

The score is `|cap|`, with no partial credit. A returned set is only scored if it is a *valid*
cap, checked by the standard incremental procedure: index each vector as `idx(v) = Σ_j v_j ·
3^{n−1−j}`, build the set one point at a time carrying a boolean `is_blocked` array over all
`3^n` indices, and when admitting a point `p`, mark blocked the closing point `r = (−p − q) mod
3` of the line `{p, q, r}` for every earlier point `q`. The set is a valid cap iff no point is
ever already blocked when replayed in order (and there are no duplicates). This is `O(c^2 n)`;
small instances are also cross-checked by an independent `O(c^3)` triple scan.

## Greedy construction

The standard constructive approach is *greedy admission*: maintain a running set, and repeatedly
add a vector that keeps the set a valid cap, blocking the completing point `r = (−p − q) mod 3` of
every line `{p, q, r}` the new point forms with an earlier point `q`. The order in which candidate
vectors are considered determines the result. Two simple orderings are practiced:

- **Lexicographic**: walk the vectors of `{0,1,2}^n` in index order and add each one that is still
  admissible. Deterministic, a single fixed pass.
- **Random multi-start**: shuffle the vectors and greedily admit in that random order, repeating
  over many independent shuffles and keeping the largest cap found.

Both plug into the same greedy skeleton; they differ only in how they choose which valid vector to
admit next.

## Code framework

```python
import itertools
import numpy as np

def all_vectors(n):
    return np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)

def is_cap_set(vectors):
    """Valid cap iff no three distinct points sum to 0 mod 3. O(c^2 n)."""
    if len(vectors) == 0:
        return True
    vectors = np.asarray(vectors, dtype=np.int32)
    c, n = vectors.shape
    powers = np.array([3 ** j for j in range(n - 1, -1, -1)], dtype=np.int64)
    raveled = vectors.astype(np.int64) @ powers
    if len(set(raveled.tolist())) != c:
        return False
    is_blocked = np.full(3 ** n, False, dtype=bool)
    for i, (v, idx) in enumerate(zip(vectors, raveled)):
        if is_blocked[idx]:
            return False
        if i >= 1:
            blocking = ((-vectors[:i, :] - v[None, :]) % 3).astype(np.int64) @ powers
            is_blocked[blocking] = True
        is_blocked[idx] = True
    return True
```

The constructor returns a set of distinct vectors in `{0,1,2}^n`; the harness verifies the cap
property and reports the size.
