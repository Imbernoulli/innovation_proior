# Well Field: Backtracking a Buried Contaminant Plume

## Problem

An aquifer is modeled as an `N x N` grid of unit cells. Field records show the leak
comes from **at most 3** distinct joints (a sparse set of source cells), each releasing
contaminant starting at time 0. Under a steady wind/flow `(vx, vy)` and isotropic
diffusivity `D`, an instantaneous unit release from cell center `(x0, y0)` produces, at
point `(x, y)` after elapsed time `t`, the concentration (2D advection-diffusion Green's
function):

```
G(x0,y0,x,y,t) = exp( -((x - x0 - vx*t)^2 + (y - y0 - vy*t)^2) / (4*D*t) ) / (4*pi*D*t)
```

Concentration is linear and additive: the reading at any point/time is
`sum_over_cells( rate_c * G(cell_c, point, t) )`. You are given noisy readings from `K`
**visible** monitoring wells, each read at `MT` fixed times, plus a mass budget (an
upper bound on total release rate, from independent flow-meter records). Your job:
output an estimated release-rate map over every grid cell.

**Why this is hard:** `K` is small and the grid has many more cells than `K*MT`
readings, so the visible system is severely underdetermined and its columns (nearby
cells) are highly correlated under the smoothing kernel `G`. Many very different source
maps -- including smeared multi-cell blobs -- fit the visible wells almost exactly.
Least-squares fitting alone cannot tell them apart; only committing to the sparsity
prior, even at the cost of a slightly worse fit to the visible data, recovers the true
source and predicts correctly *away* from the wells you were shown.

## Input (stdin)

```
testId N K MT S_MAX
D vx vy
t_1 t_2 ... t_MT
B_mass
row_1 col_1 r_1_1 ... r_1_MT
...
row_K col_K r_K_1 ... r_K_MT
```
`N` grid side (`5<=N<=11`); `K` visible wells; `MT=3` observation times; `S_MAX=3` is the
sparsity prior; each well line gives its grid cell and its `MT` noisy nonnegative
concentration readings.

## Output (stdout)

Exactly `M = N*N` whitespace-separated nonnegative numbers: the estimated release rate
for grid cell `(i,j)`, in row-major order (`index = i*N+j`).

## Feasibility

Rejected (`Ratio: 0.0`) unless: exactly `M` finite numeric tokens; every value `>= 0`
(a tolerance of `1e-6` below zero is clamped); and the total `sum(rates) <=
1.10 * B_mass`.

## Objective

Maximize recovery quality `F = 0.8*L + 0.2*H`:
- `L` (localization): normalize your rate map to a probability distribution over grid
  cells and compute its exact optimal-transport (earth-mover) distance `d` to the true
  (unpublished) source distribution, Euclidean grid-cell cost. `L = exp(-d / 1.5)` -- a
  spike on the true cell(s) scores near 1; a distribution smeared far from them decays
  fast.
- `H` (held-out fit): the checker forward-simulates your rate map at monitoring wells
  you were **never shown**, and compares to the true concentrations there:
  `H = exp(-relerr)`, `relerr` = relative L2 error over the held-out wells and times.

## Scoring

The checker also builds its own baseline `B`: release the whole mass budget **uniformly
over every grid cell** (scored the same way `F` is), then reports
```
Ratio = min(1.0, 0.1 * F / B)
```
so the uniform-spread baseline scores exactly 0.1, and 10x-better recovery caps at 1.0.
Deterministic and reproducible: the true source, held-out wells, and all noise are fixed
functions of `testId` alone (no wall-clock, no external randomness).

## Constraints

`5<=N<=11`, `4<=K<=6`, `MT=3`, `S_MAX=3`, time limit 5s.

## Example

Toy illustration (not an actual test case): a `2x2` grid, one true source at cell
`(0,0)` with rate 10, `D=1, vx=vy=0`, one reading time `t=1`. Then
`G = exp(-((x-0.5)^2+(y-0.5)^2)) / (4*pi)`. A guess that puts all mass at `(0,0)` has
`L` near 1 (zero transport distance) and, if it also reproduces the readings correctly,
`H` near 1, so `F` near 1 -- far above the uniform-spread baseline's `F`, which spreads
mass (and thus `L`) across all 4 cells.
