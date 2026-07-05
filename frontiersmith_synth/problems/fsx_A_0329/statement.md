# Formation-Flying Telescope Constellation: Uniform 3D Baseline Coverage

## Problem
A next-generation space observatory flies `n` free-floating apertures in formation. Each aperture is
assigned a normalized configuration triple in the unit cube `[0,1]^3`: two coordinates fix the
projected baseline on the sky plane and the third fixes the optical-delay (altitude) slot. To keep
the synthesized point-spread function clean, mission planning wants the apertures to fill the cube as
**uniformly** as possible, so that **every axis-aligned box anchored at the corner** `(0,0,0)` holds a
fraction of apertures as close as possible to its volume. This "corner-anchored uniformity" is exactly
the classical **L-infinity star discrepancy** of the point set: the worst-case gap, over all boxes
`[0,a1) x [0,a2) x [0,a3)` and their closed counterparts with `a1,a2,a3 in [0,1]`, between the fraction
of apertures inside the box and the box's volume.

Place all `n` apertures to make the star discrepancy **as small as possible**.

## Input (stdin)
One line with two integers:
```
n d
```
`n` = number of apertures to place, `d` = configuration dimension (always `d = 3` in this problem).

## Output (stdout)
Exactly `n` lines. Line `i` gives the configuration triple of aperture `i`:
```
x y z
```
with `0 <= x,y,z <= 1` (floats). You must output exactly `n` points, each with 3 coordinates.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- Exactly `n` points are present, each with 3 finite coordinates (no `nan`/`inf`).
- Every coordinate lies in `[-tol, 1+tol]` (coordinates are then clamped into `[0,1]`).

Any violation scores `Ratio: 0.0`.

## Objective (minimize)
Let `P = {p_1, ..., p_n}` be your points. For a corner `q = (q1,q2,q3)` define the volume
`V(q) = q1*q2*q3`, the closed count `Nc(q) = #{i : p_i,j <= q_j for all j}`, and the open count
`No(q) = #{i : p_i,j < q_j for all j}`. The star discrepancy is
```
D* = max over q of  max( Nc(q)/n - V(q) ,  V(q) - No(q)/n ).
```
This supremum is attained on the finite grid of corners whose per-axis coordinates are the point
coordinates in that axis (plus `1`), so the checker computes `D* = F` exactly. Smaller `F` is better.

## Scoring
The checker builds an internal trivial baseline `B`: the star discrepancy of the `n`-point
**main-diagonal** set `p_i = ((i+0.5)/n, (i+0.5)/n, (i+0.5)/n)`. With `F` your discrepancy,
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
so a solution matching the diagonal baseline scores about `0.1`, and one with ten times smaller
discrepancy caps at `1.0`. There is no known closed-form optimum for finite `n` in 3D; the landscape
is open-ended and rewards better constructions and real search.

## Constraints
`8 <= n <= 30`, `d = 3`. The exact discrepancy computation and scoring run well within the limit.

## Example
This is an ILLUSTRATIVE score walkthrough, not an optimal placement. Take `n = 2` and the diagonal set
`(0.25,0.25,0.25),(0.75,0.75,0.75)`. Consider the closed corner `q = (0.75,0.75,0.75)`: it contains
`Nc = 2` points with volume `V = 0.421875`, giving `Nc/n - V = 1.0 - 0.421875 = 0.578125`. No corner
does worse, so this defines `B` for the diagonal at that size. A well-spread lattice or Hammersley set
scatters the two points off the diagonal (e.g. one low-x/high-z and one high-x/low-z), which lowers the
worst-corner gap and therefore scores above `0.1`.
