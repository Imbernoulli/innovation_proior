I want a large cap set in `F_3^n` — a subset of the length-`n` vectors over the integers mod 3 with
no line, meaning no three distinct points `a, b, c` satisfy `a + b + c ≡ 0 (mod 3)`. Before I get
clever I want a floor: the simplest rule that is *guaranteed* to produce a valid cap, so every later
idea has a concrete number to beat. The constraint that defines the problem is local in a way I can
exploit. Over `F_3` the line through two points `a, b` is exactly `{a, b, −a−b}`, so any two points
determine a single third point that would complete a forbidden triple. That hands me an incremental
rule that can never go wrong: walk through the points of `F_3^n` in some fixed order, and admit a
point only if it is not the completion of a line through two points I have already admitted. Whatever
order I use, the result is a cap by construction, because each point was checked against all its
predecessors at the moment it joined. So validity is free — the real question is which order to walk,
and how good the cap it yields is.

The most naive order, requiring no thought, is lexicographic: list the vectors as if counting in base
3 and greedily take each one if it keeps the set cap-valid. I expect this to be rigid. Watch the last
coordinate: I take `0…00` and `0…01`, but `0…02` would complete the line `{0…00, 0…01, 0…02}` (their
sum is `0`), so it is rejected — already at `n = 1` the cap is `{0, 1}`, size `2`. That
two-symbols-per-coordinate pattern propagates as I lift to larger `n`, and I expect the cap to come out
at exactly `2^n`: `2, 4, 8, 16, 32, 64, 128`. That matches the optimum only at `n = 1, 2` (where `2^n`
*is* the optimum) and falls progressively short afterward — `8` vs `9` at `n = 3`, `16` vs `20`, and a
gap that widens because `2^n` grows far slower than the true cap growth rate near `2.756^n`.

Why so weak? Because lexicographic order commits early and locally. It grabs a dense cluster of
low-index points first, and those points' lines block a structured swath of the remaining space,
funnelling the greedy fill into the rigid power-of-two pattern. The counting order has nothing to do
with the geometry of `F_3^n` — it is an artifact of how I enumerate tuples — and the greedy rule cannot
look ahead, reconsider, or prefer a point that blocks few future candidates over one that blocks many.
A cap that reaches the optimum has to place its points so their lines fall outside the cap as
efficiently as possible, and a fixed blind walk has no mechanism to do that.

So the method I propose is **greedy lexicographic**, and I propose it precisely as the floor of the
ladder. It is correct by construction, deterministic, parameter-free, and reproducible, and it pins
down the trivial `2^n` baseline that every smarter ordering must beat. Its whole value is to make the
gap to the optimum concrete and to isolate "the ordering" as the single lever the rest of the work
must pull — once I see it lands on exactly `2^n`, I know the entire deficit is attributable to the
geometry-blind order, and the cheapest next move is to stop trusting any one order and try many. I run
every returned cap through the incremental verifier (and an independent triple scan at small `n`) so no
size is ever taken on faith; here the sizes come out exactly `2, 4, 8, 16, 32, 64, 128` for `n = 1..7`,
all valid.

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
            blocking = ((-capset - v[None, :]) % 3).astype(np.int64) @ powers
            is_blocked[blocking] = True
        is_blocked[idx] = True
        cap.append(v)
        capset = np.concatenate([capset, v[None, :]], axis=0)
    return np.array(cap, dtype=np.int32)
```
