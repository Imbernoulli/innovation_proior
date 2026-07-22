# Ringless Radial Tessera

## Problem

A mosaicist is tiling an `n x n` floor to approximate a radial gradient
seen from above. You are given a **fixed multiset** of `c` tile colors
(the kiln already fired an exact count of each color; nothing more, nothing
less) and must place every tile into exactly one grid cell.

Cell `(i, j)` (0-indexed, row `i`, column `j`) has value `v[color]`
determined by the color placed there, and a target intensity

```
target(i, j) = Tcenter + (Tedge - Tcenter) * dist(i, j) / rmax
```

where `dist(i, j)` is the Euclidean distance from `(i, j)` to the grid
center `((n-1)/2, (n-1)/2)`, and `rmax` is the distance from the center to
a corner. This is a smooth radial gradient: `target` depends only on how
far a cell is from the center.

The floor also has a building code: no `K+1` (or more) tiles of the same
color may sit consecutively in a single row or a single column
(orthogonal runs only; diagonals are unrestricted).

## Input

```
n c K
v_1 v_2 ... v_c            (strictly increasing tile values)
cnt_1 cnt_2 ... cnt_c       (exact tile counts, sum = n*n)
Tcenter Tedge
```

## Output

`n` lines of `n` integers each (row-major): the color index (`1..c`)
placed at every cell.

## Feasibility

1. Every value is a color index in `[1, c]`.
2. The multiset of printed colors matches `cnt[]` **exactly** (same count
   per color, not just the same total).
3. No orthogonal run (within a row, or within a column) of the same color
   exceeds length `K`.

Any violation scores 0.

## Objective (maximize)

Let `vrange = v[c] - v[1]`. Define, over all cells,

```
fidelity = sum_{cells} ( vrange - |v[color(cell)] - target(cell)| )
```

A **ring** is the set of cells sharing the same nearest-integer distance
to the center; within a ring, order cells by angle around the center
(circular order). A **transition** is a pair of angularly-adjacent
same-ring cells holding different colors. Let `dispersion` be the total
transition count across all rings, normalized by the maximum possible
transition count (each ring of size `m >= 2` can contribute at most `m`
transitions).

```
Score = fidelity * (1 + BONUS * dispersion)
```

for a fixed positive constant `BONUS` (its value and the exact ring
bucketing are implemented identically in the checker and do not need to be
reverse-engineered from the statement -- what matters is the *shape*:
dispersion multiplies fidelity, so spreading whichever tiles cannot match
their target perfectly evenly around each ring is worth real points even
at a small fidelity cost). Your score is this value divided by a fixed
reference computed by the checker, capped at 1.0.

## Constraints

`5 <= n <= 60`, `3 <= c <= 10`, `1 <= K <= 4`, `0 <= v_i <= 1000`,
`sum(cnt) = n^2`. Time limit: **5s**. Memory limit: **512MB**.

## Example

`n=3, c=3, K=2`, `v = [0, 10, 20]`, `cnt = [3, 3, 3]`,
`Tcenter=20, Tedge=0` (so the center wants value 20, the corners want 0,
the edge-midpoints want about 5.86).

One feasible output:
```
1 2 1
2 3 2
1 3 3
```
This uses exactly three tiles of each color, and the longest orthogonal
run is 2 (`col 2` has `3 3` at the last two rows) -- legal since `K=2`.
The corner cell `(2,2)` and the edge cell `(2,1)` both had to take color 3
(index 2, value 20) even though it fits them poorly, because after the
"obviously correct" cells are filled, color 3's remaining stock has
nowhere else to go; a solver that thinks only in terms of "nearest color
per cell" has no way to decide *which* two cells should absorb that
mismatch, while a solver that reasons about rings can spread such
mismatches over multiple rings/angles instead of concentrating them.
