# Mountain Rescue Relays: A Uniform 3D Relay Grid

## Problem
A mountain search-and-rescue operation must blanket a rugged region with `m` portable radio
relays. The region is normalized to the unit cube `[0,1]^3`, where the three axes are
normalized **latitude**, **longitude**, and **altitude**. For coordination, every wedge of
terrain anchored at the low corner `(0,0,0)` — every box `[0,a1] x [0,a2] x [0,a3]` — should
contain a fraction of relays as close as possible to its volume. This "corner-anchored
uniformity" is exactly the classical **3D star discrepancy** of the relay set: the worst-case
gap, over all anchored boxes, between the fraction of relays inside the box and the box's volume.

Place all `m` relays to make the star discrepancy **as small as possible**.

## Input (stdin)
One line with two integers:
```
m d
```
`m` = number of relays to place, `d` = spatial dimension (always `d = 3` in this problem).

## Output (stdout)
Exactly `m` lines. Line `i` gives the coordinates of relay `i`:
```
x y z
```
with `0 <= x,y,z <= 1` (floats). You must output exactly `m` points, `3m` numbers total.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- Exactly `3m` numbers are present (`m` relays, 3 coordinates each).
- Every coordinate is finite (no `nan`/`inf`) and lies in `[-tol, 1+tol]`; in-range
  coordinates are then clamped into `[0,1]`.

Any violation scores `Ratio: 0.0`.

## Objective (minimize)
Let `P = {p_1, ..., p_m}` be your relays. For a corner `q = (q1,q2,q3)` define the volume
`V(q) = q1*q2*q3`, the closed count `Nc(q) = #{i : p_i <= q componentwise}`, and the open
count `No(q) = #{i : p_i < q componentwise}`. The star discrepancy is
```
D* = max over q of  max( Nc(q)/m - V(q) ,  V(q) - No(q)/m ).
```
The supremum is attained on the finite grid whose per-axis candidate coordinates are the relay
coordinates (plus `1`), so the checker computes `D*` **exactly**. Smaller `D* = F` is better.

## Scoring
The checker builds an internal trivial baseline `B`: the 3D star discrepancy of the `m`-relay
**main-diagonal** set `p_i = ((i+0.5)/m, (i+0.5)/m, (i+0.5)/m)`. With `F` your discrepancy,
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
so a solution matching the diagonal baseline scores about `0.1`, and one with ten-times-smaller
discrepancy caps at `1.0`. There is no known closed-form optimum for these sizes in 3D, so the
best achievable score is genuinely open — different lattice families and local refinements win on
different instance sizes.

## Constraints
`10 <= m <= 30`, `d = 3`. The exact discrepancy computation and scoring run well within the limit.

## Example
Illustrative only. Take `m = 4` relays on the diagonal:
`(0.125,0.125,0.125), (0.375,0.375,0.375), (0.625,0.625,0.625), (0.875,0.875,0.875)`.
Its worst anchored corner sits near `q = (0.625,0.625,0.625)`, which contains `Nc = 3` relays
with volume `V = 0.625^3 = 0.244`, giving `Nc/m - V = 0.75 - 0.244 = 0.506`. This diagonal value
defines the baseline `B` for `m = 4`. A well-spread lattice — for example a rank-1 Korobov set
whose three coordinates cycle through distinct residues — pushes `D*` far lower and therefore
scores well above `0.1`.
