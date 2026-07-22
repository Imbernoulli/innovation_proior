# Curvature-Tracking Gores: Sewing a Flat Skin onto a Curved Drum

## Problem

A doubly-curved "drum skin" is given as a triangulated heightfield: a grid
of `(R+1) x (C+1)` vertices, vertex `(i,j)` sitting at 3D point
`(i, j, h(i,j))`. Each unit grid cell `(i,j)` (`0<=i<R`, `0<=j<C`) is split
into two triangles by the diagonal `(i,j)-(i+1,j+1)`:
`T_low = {(i,j),(i+1,j),(i+1,j+1)}` and `T_high = {(i,j),(i+1,j+1),(i,j+1)}`,
enumerated in that order, cell by cell, row-major (`i` outer, `j` inner) --
this fixes triangle index `t = 2*(i*C+j)` for `T_low` and `t+1` for
`T_high`. There are `N = 2*R*C` triangles total.

You must partition the `N` triangles into **panels**: assign each triangle a
non-negative integer panel id such that every panel's triangles form one
edge-connected region (a real, cuttable patch of skin, not scattered
pieces). Cutting between two differently-labeled triangles across a shared
edge costs the 3D Euclidean length of that edge; the **total seam length**
(summed over ALL edges whose two triangles differ in panel id) must not
exceed a given budget `B`.

A vertex `(i,j)` with `1<=i<=R-1` and `1<=j<=C-1` has a complete ring of 6
triangles around it and a well-defined discrete Gaussian curvature (angle
defect) `K(i,j) = 2*pi - (sum of the 6 triangle-corner angles at that
vertex)`, computed directly from the 3D triangle geometry. If ALL 6 of a
vertex's triangles share one panel id, that vertex is fully interior to a
panel; by Gauss's Theorema Egregium a panel containing it cannot be
isometrically flattened into the plane without strain proportional to
`|K(i,j)|`. If at least one of its 6 triangles carries a different panel
id, the vertex sits on a seam and is "relieved" -- it contributes nothing.
Domain-boundary vertices (`i` or `j` at 0 or R/C) are always free and never
contribute. Seams are therefore a **budgeted resource for absorbing
curvature**: they should track where curvature actually concentrates, not
follow any fixed symmetric pattern (equal meridian gores are only optimal
for a surface of revolution).

## Input (stdin)
```
R C
B
h[0][0] h[0][1] ... h[0][C]
h[1][0] ...
...
h[R][0] ... h[R][C]
```
`R,C` the grid dimensions, `B` the seam budget, then `R+1` rows of `C+1`
floats giving the height field.

## Output (stdout)
Exactly `N = 2*R*C` non-negative integers (any whitespace-separated
layout), the panel id of triangle `t` for `t = 0..N-1` in the canonical
order defined above.

## Feasibility
- Exactly `N` tokens, each a finite integer in `[0, N-1]`.
- Every panel id that is used must label an edge-connected set of
  triangles (BFS/union-find over shared edges within the same id).
- Total seam length (sum of 3D edge lengths where the two adjacent
  triangles differ in id) must be `<= B` (tolerance `1e-6`).
Any violation scores `Ratio: 0.0`.

## Objective (minimize)
`F = max` over full-interior vertices `(i,j)` (`1<=i<=R-1`, `1<=j<=C-1`)
of `|K(i,j)|`, counting only vertices whose full 6-triangle fan shares one
panel id (relieved vertices contribute 0).

## Scoring
The checker also evaluates the single-panel construction (no cuts at all,
seam length 0, always feasible) to get its own baseline `F_base` -- the
worst-case unrelieved maximum curvature of the whole surface. Your score is
`min(1.0, 0.1 * F_base / F)`, printed as `Ratio: <value>`. Cutting seams
that relieve the true curvature peaks lowers `F`; wasting the budget on
seams that don't reach any peak leaves `F` near `F_base`.

## Constraints
`6 <= R,C <= 28`, `2 <= B`. Time limit 5s, memory 512MB.

## Example (illustrative form only, not a real test case)
A `4x4` cell skin (`R=C=4`) with one bump near vertex `(2,2)` giving
`K(2,2) = 1.1`, all other interior vertices near-flat. `F_base = 1.1`
(single panel). If a submission spends part of its budget on seams around
`(2,2)` (any one of its 6 incident triangles differing from its neighbor's
id relieves it) and no other vertex has larger `|K|`, then `F` drops to the
next-largest interior `|K|`, say `0.05`, giving
`Ratio = min(1.0, 0.1*1.1/0.05) = 1.0` for this toy illustration -- actual
test instances have several competing peaks of comparable size so budget
allocation is a real trade-off, not a single free win.
