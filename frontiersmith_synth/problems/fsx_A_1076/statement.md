# Fold-and-Cut Snowflake: Orbit-Aware Hole Design

## Problem
A square sheet of paper is represented as an `N x N` grid of unit cells (`N` even), rows
and columns indexed `0..N-1`. The sheet has already been folded according to a fixed
**fold arrangement** of type `t`, which determines a group `G` of symmetries of the grid:

- `t = 2`: `G = {identity, mirror across the vertical center line}` (fold once).
- `t = 4`: `G = {identity, vertical mirror, horizontal mirror, 180-degree rotation}`
  (fold in half twice).
- `t = 8`: the full symmetry group of the square (fold in half, fold in half again, then
  fold along the diagonal) -- identity, the two mirrors, the 180-degree rotation, the two
  diagonal mirrors, and the two 90-degree rotations. `|G| = 8`.

You make a set of **cut marks**: cells you mark for cutting while the paper is folded.
Because the paper is folded, cutting through the stack at a marked cell `(r,c)` cuts every
layer, i.e. it cuts the whole **orbit** `{g(r,c) : g in G}` of that cell under `G` once the
sheet is unfolded. The cells removed from the sheet are the union of the orbits of all
your marks.

**Feasibility.** You may issue at most `V` marks, all at distinct cells of the grid, each
in bounds. After computing the union of orbits and removing those cells, the remaining
paper (the complement in the `N x N` grid) must be a single 4-connected piece -- if it is
empty or splits into two or more pieces, the output scores `0`.

## Input (stdin)
```
N t V
```
`N`: grid side length (even). `t`: fold type, one of `2, 4, 8`. `V`: maximum number of
cut marks you may issue.

## Output (stdout)
```
K
r_1 c_1
...
r_K c_K
```
`K <= V` distinct marks, each `0 <= r,c < N`.

## Objective
Let `removed` be the union of orbits of your marks, and `area_ratio = |removed| / N^2`.
Score component 1 rewards `area_ratio` being close to `1/3`:
`area_score = max(0, 1 - |area_ratio - 1/3| / (1/3))`.

Among the connected components of `removed`, call a component an **interior hole** if none
of its cells touch the sheet's outer boundary (`r` or `c` equal to `0` or `N-1`); boundary-
touching components are edge notches, not holes. Two interior holes are **incongruent** if
neither is a translated/rotated/reflected copy of the other (checked exactly, no
tolerance). Let `holes` be the number of pairwise-incongruent interior hole shapes, and let
`D = max(2, min(8, V // 6))`. Score component 2 rewards shape diversity:
`diversity_score = min(1, holes / D)`.

Maximize `F = 0.5 * area_score + 0.5 * diversity_score`.

Note the trap: marking many cells that only ever grow **one** connected cut region gives at
most one shape class among its (possibly several, but mutually congruent) unfolded copies --
`holes` stays at `1` however large that region is, since every image of one shape under an
isometry of `G` is congruent to it. Reaching `holes >= D` requires several **differently
shaped** cut regions, and placing them at varied clearances from the mirror/diagonal axes
of `G` is what keeps their orbit images from merging into each other or into a single
symmetric blob.

## Scoring
The checker builds its own trivial reference construction: a short run of a few isolated
single-cell marks placed generically in one row (shrunk automatically if that row would
disconnect the sheet for the given `t`), giving objective value `B` (always positive).
With maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the reference mark scores `Ratio = 0.1`; scoring `10x` the reference caps at
`Ratio = 1.0`.

## Constraints
- `16 <= N <= 40`, `t in {2,4,8}`, `V` given in the input (roughly `1.2 * N^2 / (3t)`).
- Time limit 5s, memory 512m.

## Example
`N=16, t=2, V=10`. Marking `(6,3)` alone: `G={id, (r,15-c)}`, orbit `{(6,3),(6,12)}`,
`removed = {(6,3),(6,12)}`, both interior, both single cells so congruent to each other:
`holes = 1`. `area_ratio = 2/256`, far below `1/3`, so `area_score` is small and
`diversity_score = min(1, 1/D)` is also small -- this is illustrative only, not the
reference construction.
