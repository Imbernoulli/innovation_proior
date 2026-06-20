# Context: Heilbronn triangle problem at n=11, structured baseline

## Research question

Place `n = 11` points in the closed unit square `[0,1]^2` so that the **minimum** triangle area —
the smallest area over all `C(11,3) = 165` triples — is as **large** as possible. The designed
object is a constructor that emits one concrete list of 11 points, scored by a single number: the
exact minimum triangle area of that set (higher is better). One near-collinear trio makes a sliver
of almost-zero area and, being the minimum, sets the whole score, so the task is to spread points so
no three are ever nearly collinear.

This rung asks the narrowest version of the question: what is the best *structured, parameter-free*
configuration I can write down in closed form, with no search at all? It establishes the floor that
every later searched method must beat.

## How the score is defined

```
score(P) = min over all triples (i,j,k) of  |(b-a) x (c-a)| / 2
```
over all 165 triples, with all points required to lie in `[0,1]^2`. A single collinear triple gives
area 0 and zeroes the score.

## Record and yardstick

The unit-square Heilbronn record at `n = 11` is `Δ(11) = 1/27 = 0.037037` (Michael Goldberg, 1972;
conjectured optimal; tabulated on Erich Friedman's packing pages). Neighboring records: `Δ(10) ≈
0.0465`, `Δ(12) ≈ 0.0326` (Comellas–Yebra), `Δ(13) ≈ 0.0270` (Karpov). A structured baseline is
expected to reach roughly half the record.

## The editable interface

One editable function, `construct() -> (11,2) array in [0,1]^2`. The evaluator (exact min over 165
triples, with an in-square check) and `n = 11` are fixed.

```python
import numpy as np
from itertools import combinations

N = 11

def min_triangle_area(P):
    best = np.inf
    for i, j, k in combinations(range(len(P)), 3):
        a, b, c = P[i], P[j], P[k]
        area = abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])) / 2.0
        best = min(best, area)
    return best
```
