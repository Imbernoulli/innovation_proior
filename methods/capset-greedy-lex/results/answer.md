# Greedy lexicographic cap set, distilled

Greedy lexicographic is the simplest correct-by-construction cap-set builder: walk the `3^n`
vectors of `F_3^n` in fixed counting (base-3) order and admit each one iff it does not complete
a line with two already-admitted points. Over `F_3` every pair of points has a unique completing
third point `r = (−a − b) mod 3`, so admitting a point and blocking the completing point of every
line it forms keeps the running set a cap at all times. The result is a valid cap for any walk
order; lexicographic is the parameter-free default and the floor of the ladder.

## Problem it solves

Construct a large cap set in `F_3^n` — a subset with no three distinct points summing to `0 (mod
3)` — returning a *verified* valid cap, scored by its size. This method supplies the trivial
baseline every smarter constructor is measured against.

## Key idea

Validity is free: each admitted point is checked against all predecessors at admission, so the
output is a cap regardless of order. The only lever is the *order*, and lexicographic is the most
naive: deterministic, geometry-blind. It commits early and locally — grabbing a dense cluster of
low-index points whose lines block a structured swath of the space — and so lands on a rigid
power-of-two pattern.

## Why these choices

The point of starting here is to fix the floor and isolate "the ordering" as the lever the rest
of the ladder pulls. Lexicographic greedy is correct, parameter-free, and reproducible, and it
reaches exactly `2^n` at every `n` — optimal only at `n = 1, 2`, with a gap to the optimum that
widens monotonically as `n` grows (because `2^n` grows far slower than the true `~2.756^n`). All
of that deficit is attributable to the geometry-blind order, motivating multi-start and then
structured/evolved priorities.

## Hyperparameters / contract

None. Deterministic (no seed). Output is a list of distinct vectors in `{0,1,2}^n` satisfying the
cap property; cost `O(|cap|^2 · n)` to build.

## Measured results

Cap sizes (all verified, `n ≤ 6` also brute-checked): `n=1:2, 2:4, 3:8, 4:16, 5:32, 6:64,
7:128` — exactly `2^n`. Matches the optimum only at `n = 1, 2`.

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
