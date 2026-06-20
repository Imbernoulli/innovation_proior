## Research question

Place `n = 11` points anywhere in the closed unit square `[0,1]^2` so that the **smallest**
triangle they form — the minimum area over all `C(11,3) = 165` triples — is as **large** as
possible. The single thing being designed is a **constructor**: a program that emits one concrete
list of `11` points in the square, and it is scored by one number, the minimum triangle area of
that point set. Nothing about the harness is learned or stochastic at evaluation time — the
constructor's output is a fixed array of coordinates, its min-triangle-area is computed exactly
over all triples, and that real number is the whole result. Higher is better.

This is the **Heilbronn triangle problem** in its max-min form. It is the geometric twin of a
hard combinatorial packing question: you are trying to spread points out so evenly that no three
of them are ever nearly collinear, because three nearly-collinear points make a sliver of almost
zero area and that sliver, being the *minimum*, is your score. The tension is global — moving one
point to fix one bad triangle can flatten three others — which is exactly what makes the landscape
rugged and the problem a natural target for search.

## How the score is defined

The score is the exact minimum triangle area:

```
score(P) = min over all triples (i,j,k) of  |cross(P_i, P_j, P_k)| / 2,
           cross(a,b,c) = (b_x - a_x)(c_y - a_y) - (c_x - a_x)(b_y - a_y),
```

evaluated over every one of the `165` triples, with the hard constraint that all `11` points lie
in `[0,1]^2`. There is no normalization gimmick and no partial credit: a configuration with even
one collinear triple scores exactly `0`, and the only way to raise the score is to genuinely
produce a point set whose worst triangle is bigger.

`n = 11` is chosen deliberately. It is small enough that the exact evaluator over `165` triples
is cheap enough to run inside a simulated-annealing inner loop, and large enough that the optimum
is a genuine open-problem record rather than something solvable by hand. Crucially, the
best-known value at `n = 11` has a clean closed form, which gives an unusually sharp yardstick.

## The record this ladder is read against

The unit-square Heilbronn records for small `n` are tabulated in Erich Friedman's packing pages
and Goldberg's / Comellas–Yebra's work. For `n = 11` the best-known minimum area is

```
Δ(11) = 1/27 = 0.037037...,
```

found by **Michael Goldberg (1972)** and believed (though not proven) optimal. The construction
is highly structured — points sitting at rational coordinates that are multiples of `1/3` and
`2/9` along the boundary and a central lattice — which is why the record is a clean fraction
rather than a messy decimal. Neighboring records, for orientation: `Δ(10) ≈ 0.0465` (Comellas–
Yebra), `Δ(12) ≈ 0.0326` (Comellas–Yebra 2001), `Δ(13) ≈ 0.0270` (Karpov 2011). The headline
numbers to keep in view at `n = 11`:

| Reference point | min triangle area | as fraction of record |
|---|---|---|
| **Goldberg record (1972, conjectured optimal)** | **0.037037 = 1/27** | **1.000** |
| Structured baseline (regular 11-gon, this scaffold) | ~0.0215 | ~0.58 |
| Uniform random point set (typical) | ~0.001–0.005 | ~0.03–0.13 |

A related note on the state of the art: in 2025 DeepMind's **AlphaEvolve** (arXiv:2506.13131,
github.com/google-deepmind/alphaevolve_results) applied an LLM-driven evolutionary search to the
Heilbronn problem and improved the records for the *unit-area triangle* container (`n = 11`, from
`0.036` to `> 0.0365`) and the *unit-area convex region* container (`n = 13, 14`). Those are
sibling containers, not the unit square, so they do not move the `1/27` square record — but they
confirm the central methodological fact this ladder relies on: the natural tool for Heilbronn-type
point-configuration records is **stochastic / evolutionary search with a local-refinement polish**,
not a closed-form construction. The endpoint rung here is the small-scale, single-machine analogue
of that: heavy simulated annealing to find the right basin, then a soft-min gradient polish to
snap onto the exact optimum.

## Prior art before the first rung

- **Heilbronn's conjecture and the asymptotic bounds.** Heilbronn conjectured `Δ(n) = O(1/n^2)`.
  Roth (1951) and later Komlós–Pintz–Szemerédi proved upper bounds, and a `Ω(log n / n^2)` lower
  bound is known by a probabilistic deletion argument. *Gap here:* these are asymptotic
  order-of-magnitude statements; they say nothing about the exact constant at a specific small
  `n`, and at `n = 11` the record is a hand-found rational, not the output of any asymptotic
  formula.
- **Goldberg (1972) structured constructions.** Goldberg gave explicit small-`n` configurations,
  including the `1/27` arrangement at `n = 11`, by placing points at rational coordinates with a
  symmetric layout. *Gap:* these are clever hand designs for specific `n`; reproducing one from
  scratch requires either guessing the symmetry or searching, and there is no construction that
  generalizes to give the optimum at arbitrary `n`.
- **Comellas–Yebra (2001) and Karpov / Beyleveld computer search.** The records for `n = 10, 12,
  13, 14, 15, 16` come from large-scale numerical optimization — multi-start local search and
  simulated annealing on point coordinates. *Gap:* these are the output of dedicated search runs;
  the published numbers are conjectured records, not proven optima, which is exactly what leaves
  room for a discovery ladder that climbs toward them.
- **AlphaEvolve (2025) evolutionary search.** LLM-guided program evolution improved Heilbronn
  records for the triangle and convex-region containers. *Gap:* it targets sibling containers
  (unit-area triangle / convex region), uses far more compute, and does not address the unit
  square at `n = 11`; but it establishes search-plus-polish as the right shape of method.

## The fixed substrate

The harness is a thin, deterministic evaluator. It calls the constructor once, receives an
`11 × 2` array of coordinates, checks that every point lies in `[0,1]^2`, computes the minimum
triangle area **exactly** over all `165` triples (a cross-product area for each triple, then the
minimum), and returns that value. There is no held-out set and nothing to overfit: the score is a
deterministic function of the returned points. Floating-point is fine — coordinates are reals —
but the area is the exact min over all triples, so a configuration cannot hide a sliver triangle.

## The editable interface

Exactly one function is editable: `construct()`, returning the `11 × 2` array of points in the
unit square. Every rung on the ladder is a different body for it. The evaluator and the value
`n = 11` are frozen.

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

Every valid output must satisfy: shape `(11, 2)`; every coordinate in `[0,1]`. There are no other
constraints — the constructor is free to return any points, structured or searched.

## Evaluation settings

A single deterministic instance: `n = 11`, scored by the exact min-triangle-area above. Because a
constructor may be randomized internally, the run is fixed to a stated seed so the reported number
is reproducible; the harness reports the exact min-area of the *returned* points. The fixed
yardstick every rung is read against is the Goldberg record `1/27 = 0.037037`, and the honest
measure of a rung is how close its genuine measured min-area comes to that fraction.
