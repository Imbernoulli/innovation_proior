# Depth-Blind No More: Recovering a Buried Density Layer From Surface Gravity

## Problem

A subsurface cross-section is discretized into a grid of `nx` columns (`c = 0..nx-1`)
by `nz` depth layers (`z = 1..nz`, `z=1` shallowest, `z=nz` deepest). Cell `(c,z)`
carries a density contrast `rho[z][c] >= 0`. For computing its effect at the surface,
each cell acts as a point mass at its center. A station sitting directly above column
`cs` reads the sum, over every cell, of that cell's vertical-gravity contribution:

```
K(dc, z) = KGAIN * depth(z) / (dc^2 * CELL_W^2 + depth(z)^2)^1.5,   dc = cs - c
depth(z) = (z - 0.5) * LAYER_H
reading(cs) = sum over all (c,z) of rho[z][c] * K(cs - c, z)
```

with fixed constants `CELL_W = LAYER_H = 2.0`, `KGAIN = 100.0`. `K` decays roughly as
`1/depth(z)^2` directly above a cell -- shallow cells are far more "sensitive" than deep
ones, so a small mass placed shallow and a much bigger mass placed deep can look similar
at the surface.

A single hidden body -- one contiguous span of columns `[c_lo, c_hi]` at one depth layer
`z_true`, uniform density `rho_true` -- produces the true field. Stations sit above every
column; you are given the noisy readings at the **even** columns only (a "training"
transect). The **odd** columns are held out and never shown; they are still used for
grading.

## Input (stdin)
```
testId nx nz
RHO_MAX MASS_MAX
ns_given
c_1 reading_1
...
c_ns_given reading_ns_given
```
`c_i` are even column indices in `[0, nx-1]`; `reading_i` is the noisy surface reading
above that column. `RHO_MAX` bounds every cell; `MASS_MAX` bounds the total density you
may use.

## Output (stdout)
`nz` lines, each with `nx` numbers: row `z` (shallowest first) is `rho[z][0..nx-1]`.

## Feasibility
- Exactly `nz * nx` finite numeric tokens.
- Every value in `[0, RHO_MAX]`.
- `sum` of all values `<= MASS_MAX` (a small numerical slack is allowed).
Any violation scores `Ratio: 0.0`.

## Objective (maximize)
Two components, each in `[0,1]`:
- **Shape match**: a cell counts as "recovered" if `rho[z][c] >= 0.4 * RHO_MAX`. `IoU` is
  the intersection-over-union of your recovered cells against the true body's cells.
- **Field match**: predict every station's reading (given AND held-out) from your grid,
  then `fitq = 1 - mean(min(1, |pred-obs| / (|pred|+|obs|+eps)))` over all stations.

`Q = 0.6 * IoU + 0.4 * fitq`. The checker also computes `Q` for its own simple reference
guess (one cell of density at the shallowest layer, placed to explain only the single
largest given reading). Your score is `min(0.92, 0.1 * Q / B)` where `B` is that
reference's `Q` -- the reference always scores `0.1`; ten times better on `Q` saturates
at the `0.92` cap (headroom is intentional).

## Why shallow guesses fail
Because `K` decays like `1/depth^2`, a shallow-only reconstruction can usually reproduce
the *numbers* at nearby stations reasonably well (there are enough free shallow cells to
locally match almost any smooth curve) -- but its recovered cells sit at `z=1`. Whenever
the true body is deeper, that recovery has **zero cell overlap** with the truth, so `IoU`
is exactly `0` no matter how good the field fit looks. Field fit alone cannot certify
you found the right depth; only the shape term can.

## Constraints
`8 <= nx <= 16`, `4 <= nz <= 7`, `RHO_MAX = 10.0`. Time limit 5s.

## Example (illustrative form only, not a real test case)
`nx=4, nz=2`. Given readings at columns `0,2`. Suppose your grid is
`[[6,6,0,0],[0,0,0,0]]` (all mass at `z=1`, columns 0-1). If the true body actually sits
at `z=2`, columns `0-1`, then `IoU = 0` (no overlap in `z`) even if the predicted
readings happen to be close to the observed ones -- the mechanism only; real tests use
larger grids with the body at varying, often deep, layers.
