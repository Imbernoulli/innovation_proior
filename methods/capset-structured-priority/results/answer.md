# Structured symmetric priority-function greedy cap set, distilled

This method replaces the random order with a deterministic *structured priority*: every vector is
scored by a hand-designed function that rewards the symmetry of `F_3^n`, and the same greedy
admission rule adds the highest-priority valid vector, blocking the completing point of every line
it forms. The priority gives the greedy fill a geometry-aligned reason to prefer one point over
another, rather than the geometry-blind counting or random order of the earlier methods.

## Problem it solves

Construct a large cap set in `F_3^n` (no three distinct points summing to `0 mod 3`), returning a
verified valid cap scored by its size. This method tests whether a *human-designed* structured
priority, plugged into the correct greedy-priority skeleton, beats brute sampling of orders.

## Key idea

Score each vector by features the known large caps exhibit: a bonus per matched **reflection pair**
`el[i] == el[n−1−i]`; a gentle preference for a coherent **weight-mod-3 layer** (Hamming weight mod
3 — the line condition `a+b+c ≡ 0` is itself about coordinate sums mod 3); and a small
coordinate-sum tie-break to total-order the vectors. Greedily admit highest-priority-first.

## Why these choices

Reflection symmetry and a coherent weight profile are natural, principled guesses about which
structure matters. But they are a *guess*: a single structured order is still one order, and if the
chosen symmetries are even slightly misaligned with the optimal cap, the result lands in the same
band as — or below — best-of-thousands random restarts. The greedy-priority *skeleton* is correct
(it is the exact machinery the strong constructions use); the *hand-designed* priority plugged into
it plateaus. That is the lesson: the priority must be the *right* one, which motivates discovering
it by search rather than hand-design.

## Hyperparameters / contract

Fixed priority weights (reflection bonus `1.0`, weight-layer bonus `0.5·(3 − w mod 3)`, sum
tie-break `0.01`). Deterministic given `n`. Output is a verified valid cap; cost `O(3^n · n)` to
score plus `O(|cap|^2 · n)` to build.

## Measured results

Cap sizes (verified, `n ≤ 6` brute-checked): `n=4:18, 5:36, 6:64, 7:138`. Beats the lexicographic
floor at `n = 4,5,7` but lands *below* random multi-start (`20, 39, 77, 147`) at every `n` —
structure is necessary but a hand guess is not sufficient.

```python
import itertools
import numpy as np

def priority(el, n):
    """Structured symmetric priority: reflection matches + weight-mod-3 layer + sum tie-break."""
    score = float(n)
    for i in range(1, n // 2):                 # reflection pairs i <-> n-1-i
        if el[i] == el[n - 1 - i]:
            score += 1.0
    w = sum(1 for e in el if e != 0)           # Hamming weight
    score += (3 - (w % 3)) * 0.5               # prefer a coherent weight-mod-3 layer
    score += 0.01 * sum(el)                    # deterministic tie-break
    return score

def construct(n):
    """Priority-function greedy: add highest-priority valid vector, block closed lines."""
    av = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)
    powers = 3 ** np.arange(n - 1, -1, -1)
    priorities = np.array([priority(tuple(int(x) for x in v), n) for v in av], dtype=float)
    capset = np.empty((0, n), dtype=np.int32)
    while np.any(priorities != -np.inf):
        k = int(np.argmax(priorities))
        v = av[None, k]
        blocking = np.einsum('cn,n->c', (-capset - v) % 3, powers).astype(np.int64)
        priorities[blocking] = -np.inf            # block completing point of each line
        priorities[k] = -np.inf
        capset = np.concatenate([capset, v], axis=0)
    return capset
```
