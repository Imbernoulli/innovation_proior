# Salmon Migration Ladder: Completing a Uniform Camera Grid

## Problem
A fisheries agency monitors a salmon migration ladder with underwater cameras. Each
camera is described by a point `(x, y)` in the unit square `[0,1]^2`:

- `x` = fraction of the ladder climbed (`0` = river mouth, `1` = spawning ground),
- `y` = fraction of the daily cycle (`0` = midnight, `1` = the next midnight).

To catch fish at **every** position and **every** time of day, the `M` cameras must
sample the `(position x time)` plane as **uniformly** as possible: the sampling
quality of a set of `M` points is its **star discrepancy** (the largest gap, over all
axis-aligned boxes anchored at the origin, between the fraction of points in the box
and the box's area). Lower is better.

The twist: `k` cameras are **already bolted in** at fixed legacy locations (given to
you and unmovable). You must choose where to install the remaining `M - k` cameras so
that the star discrepancy of the **full** set of `M` cameras is as small as possible.

This is a low-discrepancy point-set **completion** problem: the induced-box star
discrepancy is measured exactly over the whole set.

## Input (stdin)
```
M k
<k lines, each: x y>
```
- `M` — total number of cameras (fixed + new).
- `k` — number of pre-installed (fixed) cameras.
- the next `k` lines give the fixed camera coordinates, each `x y` with `0 <= x,y <= 1`.

## Output (stdout)
```
<M-k lines, each: x y>
```
Print the coordinates of the `M - k` **new** cameras you install, one `x y` pair per
line (real numbers). Print exactly `M - k` points.

## Feasibility
An output is valid iff **all** hold (tolerance `1e-6`):
- exactly `M - k` coordinate pairs are printed;
- every new camera lies in the unit square: `0 <= x <= 1`, `0 <= y <= 1`
  (values within `1e-6` of the border are clamped in).
Any violation scores `Ratio: 0.0`. (Cameras may coincide; discrepancy handles it.)

## Objective
Let `P` be the union of the `k` fixed cameras and your `M - k` new cameras
(`|P| = M`). Minimize the exact star discrepancy
```
F = max over all boxes [0,a) x [0,b) and [0,a] x [0,b] of
        | (#points of P inside the box) / M  -  a*b |,
```
evaluated exactly over the finite grid of coordinate values induced by `P` (the only
places where the discrepancy can be maximal).

## Scoring
The checker builds its own baseline `B`: complete the set by placing all `M - k` new
cameras on the **main diagonal** `((i+0.5)/(M-k), (i+0.5)/(M-k))`, union with the
fixed cameras, and take `B =` that full set's star discrepancy (deliberately poor,
since the new cameras are all concentrated on a line). With minimization
normalization:
```
sc    = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Reproducing the diagonal completion scores `Ratio = 0.1`; a completion whose full-set
discrepancy is `10x` smaller than the baseline caps at `1.0`.

## Constraints
- `24 <= M <= 88`; `k = round(0.25 * M)` fixed cameras (at least `3`).
- Fixed cameras are scattered over `[0,1]^2` (a deterministic function of the test id).
- Time limit 5s, memory 512m.

## Example
Suppose `M = 24`, `k = 6`, so you place `18` new cameras. The diagonal completion of
those 18 cameras (plus the 6 fixed) has some star discrepancy `B` and scores
`Ratio = 0.1`. If instead you place the 18 new cameras on a well-chosen rank-1 lattice
so the full 24-point set has discrepancy `F = B / 3`, then
`sc = 100 * 3 = 300`, `Ratio = 0.300`.
