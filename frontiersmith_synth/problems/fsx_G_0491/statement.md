# Bin Loading: Maximum Spherical Goods in a Storage Bin

## Problem
A warehouse stores identical spherical goods (each of radius `r = 1`) inside a rectangular
storage bin of interior dimensions `Lx x Ly x Lz`. Pack as many goods into the bin as possible:
every sphere must lie **entirely inside** the bin and no two spheres may **overlap** (they may
touch). Output the centre coordinates of the spheres you place. The score grows with the number
of spheres packed.

This is the classic hard problem of dense sphere packing in a finite container. In the bulk,
face-centred-cubic / hexagonal-close packing achieves the maximal density (Kepler), but near the
six walls the layers no longer fit cleanly, so the **optimal count for a specific finite bin is
not known in closed form** — there is real room above any lattice construction for
boundary-adapted or annealed placements.

## Input (stdin)
A single line with four real numbers:
```
Lx Ly Lz r
```
the bin interior dimensions and the sphere radius (`r = 1` in all tests).

## Output (stdout)
```
N
x_1 y_1 z_1
x_2 y_2 z_2
...
x_N y_N z_N
```
`N` = number of spheres you place, followed by `N` lines of three real coordinates each (the
sphere centres).

## Feasibility
A submission is valid iff **all** hold (checked with tolerance `1e-6`):
- `1 + 3N` numeric tokens exactly; all coordinates finite (no `nan`/`inf`).
- Containment: every centre satisfies `r <= x <= Lx - r`, `r <= y <= Ly - r`, `r <= z <= Lz - r`.
- Non-overlap: every pair of centres is at Euclidean distance `>= 2r`.

Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = N`, the number of feasibly placed spheres.

## Scoring
Let `B` be the checker's internal baseline: the count of a loose axis-aligned cubic lattice with
spacing `1.6 * (2r)` that fits the bin. The reported score is
```
Ratio = min(1000, 100 * F / B) / 1000
```
So reproducing the loose baseline scores about `0.1`; denser packings score higher, and the
normalization leaves headroom (a lattice fill lands well below `1.0`).

## Constraints
- `Lx, Ly, Lz` up to roughly `33`; `r = 1`.
- `0 <= N <= 2*10^6`.
- Deterministic scoring; time limit 5 s, memory 512 MB.

## Example (worked score)
Suppose the bin is `12 x 10.5 x 13.7` with `r = 1`. The loose baseline places
`4 x 3 x 4 = 48` spheres, so `B = 48`. A tight cubic fill (spacing `2`) places `180` spheres,
scoring `Ratio = 100 * 180 / 48 / 1000 = 0.375`. An FCC close packing seats about `252` spheres,
scoring `Ratio ≈ 0.525`. The finite-bin optimum lies somewhere above that.
