# Metro Catchment Packing: Non-Overlapping Station Reach

## Problem
A city is modelled as the unit square `[0,1] x [0,1]`. A metro line has `N` **platforms**
at fixed locations `(x_i, y_i)` inside the city (given in the input). Around each platform
you draw a circular **catchment disk** -- the walkable reach served by that station.

Because two stations' catchments must not double-count the same neighbourhood (and to keep
walking zones disjoint), the disks must be **pairwise non-overlapping** and lie **entirely
inside the city**. Every catchment disk must, of course, **contain its own platform**. Total
served reach is what the transit authority rewards, so you maximize the **sum of the disk
radii**.

Note that a disk's centre need **not** coincide with its platform: you may place the centre
anywhere, as long as the platform lies inside (or on) the disk. Relocating a centre toward
open space is exactly how you win extra radius.

## Input (stdin)
```
N
x_1 y_1
x_2 y_2
...
x_N y_N
```
`N` is the number of platforms; each `(x_i, y_i)` is a platform location strictly inside the
unit square. Platforms are distinct.

## Output (stdout)
Exactly `N` lines, one per platform **in input order**:
```
cx_i cy_i r_i
```
the centre `(cx_i, cy_i)` and radius `r_i >= 0` of platform `i`'s catchment disk (floats).

## Feasibility
All checks use tolerance `tol = 1e-6`; every number must be finite:
- `r_i >= -tol`.
- Inside the city: `cx_i - r_i >= -tol`, `cx_i + r_i <= 1 + tol`, and likewise for `cy_i`.
- Covers its platform: `(cx_i - x_i)^2 + (cy_i - y_i)^2 <= r_i^2 + tol`.
- Non-overlap: for every pair `i != j`, `dist(center_i, center_j) >= r_i + r_j - tol`.

Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F = sum_i r_i`.

## Scoring
Let `B` be the checker's internal baseline: the **uniform-radius** construction -- centre each
disk on its platform and give them all the same radius `min(smallest wall distance, half the
smallest inter-platform distance)`. This is always feasible, so `B = N * that radius`. With `F`
your feasible total radius,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the symmetric baseline scores about `0.1`, and a placement ten times better caps at `1.0`.

## Constraints
`8 <= N <= 26`. The `O(N^2)` checker runs far under the time limit for these sizes.

## Example
Two platforms at `(0.3, 0.5)` and `(0.7, 0.5)` (distance `0.4`). The uniform baseline gives
each a radius `min(0.3, 0.2) = 0.2`, so `B = 2 * 0.2 = 0.4`. A better solution slides each
centre toward its own wall -- centre `(0.2, 0.5)` radius `0.2` and centre `(0.8, 0.5)` radius
`0.2` still cover the platforms, stay inside, and remain disjoint (centres `0.6` apart >=
`0.4`). Pushing radii further by exploiting the vertical room raises `F` above `B` and lifts
`Ratio` above `0.1`. (Illustrative only -- real instances have many platforms and no such clean
symmetry.)
