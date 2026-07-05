# Lunar Habitat: Sensor Lattice for Uniform Floor Coverage

## Problem
A pressurized lunar habitat module has a square floor mapped to the unit square `[0,1]^2`.
Mission control must scatter `n` environmental sensors (dust, radiation, atmosphere) across the
floor so that **every rectangular sub-region anchored at the corner** `(0,0)` holds a fraction of
sensors as close as possible to its area. This "corner-anchored uniformity" is exactly the
classical **star discrepancy** of the sensor point set: the worst-case gap, over all boxes
`[0, a1) x [0, a2)` and `[0, a1] x [0, a2]` with `a1, a2 in [0,1]`, between the fraction of
sensors inside the box and the box's area.

Place all `n` sensors to make the star discrepancy **as small as possible**.

## Input (stdin)
One line with two integers:
```
n d
```
`n` = number of sensors to place, `d` = spatial dimension (always `d = 2` in this problem).

## Output (stdout)
Exactly `n` lines. Line `i` gives the coordinates of sensor `i`:
```
x y
```
with `0 <= x <= 1` and `0 <= y <= 1` (floats). You must output exactly `n` points.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- Exactly `n` points are present, each with 2 finite coordinates.
- Every coordinate lies in `[-tol, 1+tol]` (coordinates are then clamped into `[0,1]`).

Any violation scores `Ratio: 0.0`.

## Objective (minimize)
Let `P = {p_1, ..., p_n}` be your points. For a corner `q = (qx, qy)` define the volume
`V(q) = qx * qy`, the closed count `Nc(q) = #{i : p_i,x <= qx and p_i,y <= qy}`, and the open
count `No(q) = #{i : p_i,x < qx and p_i,y < qy}`. The star discrepancy is
```
D* = max over q of  max( Nc(q)/n - V(q) ,  V(q) - No(q)/n ).
```
This supremum is attained on the finite grid of corners whose coordinates are point coordinates
(plus `1`), so the checker computes `D*` exactly. Smaller `D* = F` is better.

## Scoring
The checker builds an internal trivial baseline `B`: the star discrepancy of the `n`-point
**diagonal** set `p_i = ((i+0.5)/n, (i+0.5)/n)`. With `F` your discrepancy,
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
so a solution matching the diagonal baseline scores about `0.1`, and one with ten times smaller
discrepancy caps at `1.0`.

## Constraints
`6 <= n <= 40`, `d = 2`. The exact discrepancy computation and scoring run well within the limit.

## Example
Take `n = 4` and the diagonal set `(0.125,0.125),(0.375,0.375),(0.625,0.625),(0.875,0.875)`.
Its worst corner is the closed corner `q = (0.625,0.625)`: it contains `Nc = 3` points with
volume `V = 0.390625`, giving `Nc/n - V = 0.75 - 0.390625 = 0.359375`, and this is the exact
`D*` of the diagonal, so it defines the baseline `B` at `n = 4`. A well-spread lattice such as
`(0.125,0.375),(0.375,0.875),(0.625,0.125),(0.875,0.625)` pushes `D*` lower and therefore scores
above `0.1`.
