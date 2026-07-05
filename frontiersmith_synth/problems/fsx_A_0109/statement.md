# Belt Survey: Minimum-Discrepancy Probe Placement

## Problem
A mining consortium has rectified its asteroid belt into a flat normalized
map, the unit square `[0,1]^2`. The belt is to be surveyed by `M` survey
stations. `K` of those stations are **already deployed** — heavy anchor rigs
bolted around a known ore vein, and they cluster in the lower-left of the map.
You control the remaining `N_free = M - K` mobile **probes**.

Your goal is to place the probes so that the **entire** set of `M` stations
(the fixed anchors together with your probes) blankets the belt as evenly as
possible. "Evenness" is measured by the exact **star discrepancy** of the
point set: the worst-case mismatch, over every axis-aligned box anchored at
the origin, between the fraction of stations inside the box and the box's
area. A low star discrepancy means no sub-region of the belt is starved of
coverage and none is over-sampled.

You cannot move the anchor rigs, so a good layout must actively **counter-
balance** their clustering.

## Input (stdin)
```
d M K
x_1 y_1
...
x_K y_K
```
- `d = 2` (the belt map is two-dimensional).
- `M` total stations, `K` pre-deployed anchor rigs, `0 <= K < M`.
- The next `K` lines give the anchor coordinates, each in `[0,1]`.
- `N_free = M - K` is the number of probes you must place.

## Output (stdout)
Exactly `N_free` lines (whitespace-separated numbers also accepted), each the
coordinates of one probe:
```
px_1 py_1
...
px_{N_free} py_{N_free}
```
Each coordinate must lie in `[0,1]`.

## Feasibility
- Exactly `N_free` probe points must be emitted.
- Every coordinate must be a finite number in `[0,1]` (a tolerance of `1e-6`
  outside the range is clamped; anything further is rejected).
Any violation scores `0`.

## Objective (minimize)
Let `P` be the union of the `K` anchors and your `N_free` probes (`|P| = M`).
Minimize the star discrepancy
```
D*(P) = sup_{ x in [0,1]^2 } | (#{ p in P : p < x }) / M  -  x_1 * x_2 |
```
computed **exactly** by the checker (the supremum is attained on the finite
grid induced by the point coordinates, so it is evaluated exactly).

## Scoring
Let `F = D*(P)` be your discrepancy and `B` the discrepancy of a trivial
corner-grid probe layout combined with the same anchors (the checker builds
`B` itself). The score is
```
sc    = min(1000, 100 * B / F)
Ratio = sc / 1000
```
So the trivial corner grid scores about `0.1`, and a layout with one-tenth the
discrepancy caps the ratio at `1.0`. Smaller discrepancy is always better.

## Constraints
- `2 <= d`, here `d = 2`.
- Instances scale up the ladder: `N_free` from ~48 up to ~192 probes, with a
  growing number of entrenched anchor rigs.
- Deterministic scoring: identical output always yields an identical score.

## Example
Suppose `d=2, M=5, K=1` with a single anchor at `(0.1, 0.1)`, so you place
`N_free = 4` probes. Emitting
```
0.7 0.7
0.7 0.2
0.2 0.7
0.5 0.5
```
spreads your probes across the upper/right regions that the lower-left anchor
neglects. The checker forms the 5-point union, computes its exact star
discrepancy `F`, compares it against the corner-grid baseline `B`, and prints
`Ratio: min(1000, 100*B/F)/1000`.
