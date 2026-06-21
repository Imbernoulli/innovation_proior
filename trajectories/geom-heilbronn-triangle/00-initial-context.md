## Research question

Place `n = 11` points in the closed unit square `[0,1]^2` so that the **minimum** triangle area over all `C(11,3) = 165` triples is as **large** as possible. The task is to design a constructor — a program that emits one concrete `11 × 2` array of coordinates in `[0,1]^2`. The score is the exact minimum triangle area:

```
score(P) = min over all triples (i,j,k) of |cross(P_i, P_j, P_k)| / 2,
           cross(a,b,c) = (b_x - a_x)(c_y - a_y) - (c_x - a_x)(b_y - a_y).
```

Higher is better; one collinear triple gives score `0`. At `n = 11` the best-known value is `Δ(11) = 1/27 = 0.037037...`, found by Michael Goldberg (1972) and conjectured optimal. The challenge is global: moving one point to fix a bad triangle can create another.

## Prior art / Background / Baselines

- **Heilbronn's conjecture and asymptotic bounds.** The conjecture `Δ(n) = O(1/n^2)` and subsequent upper/lower bounds address the rate of decay for large `n`.
- **Goldberg (1972) structured constructions.** Explicit small-`n` configurations place points at rational coordinates with symmetric layouts, including the `1/27` arrangement at `n = 11`.
- **Comellas–Yebra (2001) and Karpov computer search.** Records for `n = 10, 12, 13, ...` come from large-scale numerical optimization such as multi-start local search and simulated annealing on point coordinates.

## Fixed substrate / Code framework

The harness is deterministic: it calls `construct()` once, checks that every point lies in `[0,1]^2`, and computes the exact minimum triangle area over all `165` triples. There is no held-out set and nothing to overfit; the score is a deterministic function of the returned points.

## Editable interface

Exactly one function is editable: `construct()`, returning the `11 × 2` array of points in the unit square. Every rung on the ladder is a different body for it. The evaluator and `n = 11` are frozen.

```python
import numpy as np
from itertools import combinations

N = 11

def min_triangle_area(P):
    """Exact min over all C(n,3) triangle areas. P: (n,2) array in [0,1]^2."""
    best = np.inf
    for i, j, k in combinations(range(len(P)), 3):
        a, b, c = P[i], P[j], P[k]
        area = abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])) / 2.0
        if area < best:
            best = area
    return best

def score(P):
    P = np.asarray(P, float)
    assert np.all(P >= -1e-9) and np.all(P <= 1 + 1e-9), "points must lie in [0,1]^2"
    return min_triangle_area(P)

# ---- EDITABLE: the constructor. Default = 11 points on the inscribed circle. ----
def construct():
    ang = np.linspace(0, 2*np.pi, N, endpoint=False)
    return 0.5 + 0.5*np.column_stack([np.cos(ang), np.sin(ang)])
```

Every valid output must have shape `(11, 2)` with all coordinates in `[0,1]`.

## Evaluation settings

A single deterministic instance: `n = 11`, scored by the exact min-triangle-area above. If the constructor is randomized internally, the run is fixed to a stated seed so the reported number is reproducible. The yardstick every rung is read against is the Goldberg record `1/27 = 0.037037`.
