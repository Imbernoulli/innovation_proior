# Finding the Small Colorful Triangle: Door-to-Door Sperner Search

## Problem
Take the triangular grid of side `N`: every vertex is a triple of non-negative
integers `(x, y, z)` with `x + y + z = N`. The grid is triangulated into small
"UP" triangles `{(i,j),(i+1,j),(i,j+1)}` and "DOWN" triangles
`{(i+1,j),(i,j+1),(i+1,j+1)}`. Every vertex is pre-colored `0`, `1`, or `2`;
the coloring is given to you in full and is guaranteed to satisfy **Sperner's
boundary rule**: a vertex on the face `z=0` is never colored `2`, one on
`x=0` is never colored `0`, one on `y=0` is never colored `1`. Sperner's Lemma
then guarantees at least one small triangle is **panchromatic** (its three
vertices show all three colors) — the "colorful triangle" you must find.

Scanning every one of the `~N^2` triangles always finds one, but that is
expensive. The `z=0` edge, read from `(N,0,0)` to `(0,N,0)`, uses only colors
`{0,1}` and — because of the boundary rule — its color sequence is
guaranteed to flip from `0` to `1` **exactly once**; call the small triangle
sitting on that flip edge the *entry door*. Starting there and always
crossing into the neighbor across the triangle's *other* `{0,1}`-colored edge
(every non-panchromatic triangle you pass through has exactly one such other
edge) traces a walk that is forced to terminate at a genuine panchromatic
triangle — a constructive, parity-argument proof of Sperner's Lemma.

## Input (stdin)
Line 1: two integers `N D`. Then `(N+1)(N+2)/2` lines `x y c`, one per grid
vertex, giving its true color `c in {0,1,2}` (with `z = N - x - y` implied).
`D` is not needed to solve the problem; it is a bookkeeping constant.

## Output (stdout)
```
ANSWER x1 y1 x2 y2 x3 y3
PATH m
x0 y0 x1 y1 x2 y2         <- vertices of T_0 (one line per triangle, m lines)
...
EXTRA k                    <- optional, defaults to 0 if omitted
x y                         <- k extra grid points, one per line
```
`ANSWER` is the small triangle you claim is panchromatic. `PATH` is your
*certificate*: `T_0, ..., T_{m-1}`, each a small triangle of the grid. `EXTRA`
lets you (optionally) declare additional grid points you looked at beyond the
certificate.

## Feasibility (any violation scores 0)
- `ANSWER` is a genuine small triangle and is panchromatic under the true
  colors.
- `T_0` has an edge lying on `z=0` whose two endpoints are colored `{0,1}`
  (the entry door).
- For every `k`, `T_k` and `T_{k+1}` share exactly one edge (two vertices),
  and that shared edge's endpoints are colored `{0,1}` (a genuine "door").
- No triangle repeats in `PATH`.
- The last triangle of `PATH` is exactly `ANSWER`.

Because of the argument above, a certificate satisfying all of this — if it
exists — is *unique*: at every non-panchromatic triangle along the way there
is exactly one door besides the one you entered through, so there is no
shortcut and no way to fake a shorter one.

## Objective (maximize)
Let `OPT` be the number of distinct grid points used by the *shortest*
possible valid certificate (a fixed property of the instance). Let `USED` be
the number of distinct grid points appearing in your submitted `PATH` plus
`EXTRA` (every valid `PATH` alone already costs exactly `OPT` points, so
`USED >= OPT`). Your score is
```
Ratio = min(1.0, 0.85 * OPT / USED)
```
The unique minimal certificate (no `EXTRA`) scores `0.85`; every unnecessary
point you report on top of it lowers your score, and a large wasted search
(e.g. scanning much of the grid before you knew where to certify from) can
drive it close to `0`.

## Constraints
`N` ranges from 15 to 180 across the ladder. Coordinates and counts are
non-negative integers; time limit 5s, memory 512m.

## Example (worked score)
For a small `N=4` instance the entry door is at `T_0 = {(2,1),(3,1),(2,2)}`
(colors `0,0,1`), and the forced walk is
```
{(2,1),(3,1),(2,2)} -> {(2,1),(2,2),(1,2)} -> {(1,2),(2,1),(1,1)}
-> {(1,1),(2,1),(2,0)} -> {(1,1),(2,0),(1,0)}
```
ending at the panchromatic triangle `{(1,1),(2,0),(1,0)}` (colors `1,0,2`).
This uses `OPT = 7` distinct points. Submitting exactly this path with no
`EXTRA` gives `USED = 7 = OPT`, hence `Ratio = 0.85`. Padding `EXTRA` with,
say, 7 more (unnecessary) points would give `USED = 14` and `Ratio ≈ 0.425`.
