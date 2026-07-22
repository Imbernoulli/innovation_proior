# Escape the Cage: Extractable Support Scaffolding

## Problem
You must 3D-print a rigid voxel solid. Printing is layer-by-layer from a flat build
plate upward, so any solid voxel that floats over empty space needs **support** beneath
it. Support is printed too — and afterwards it must be **physically pulled out**. Support
buried inside an enclosed cavity cannot be removed and ruins the part.

You choose **(1)** how to lay the part on the plate (which of the 6 axis directions points
**down**) and **(2)** the set of empty cells to fill with support. You pay for support
material and are penalised for every support voxel that cannot be extracted.

## Input (stdin)
```
X Y Z LAM
<Z*Y lines, each X characters '0'/'1'>
```
Line `r` (0-indexed) after the header is layer `z = r // Y`, row `y = r % Y`; character
`x` is solid iff `'1'`. `LAM` is the extraction penalty weight.

## Orientation
`o in {0..5}` picks the **down** axis; the part is relabelled into a build frame
`(u,v,w)` with `w` the height (`w=0` = plate):
`o=0:(x,y,z)`, `o=1:(x,y,Z-1-z)`, `o=2:(x,z,y)`, `o=3:(x,z,Y-1-y)`,
`o=4:(y,z,x)`, `o=5:(y,z,X-1-x)`. Oriented sizes: `o<2 -> (X,Y,Z)`,
`o<4 -> (X,Z,Y)`, else `(Y,Z,X)`.

## Output (stdout)
```
o
K
<K lines: x y z  = a support voxel, in ORIGINAL coordinates>
```

## Feasibility (any violation => Ratio 0)
Work in the build frame for orientation `o`. Let a cell be *occupied* if it is solid or
chosen support.
- Every support voxel is an in-grid, non-solid, distinct cell.
- **Facet rule (solids):** every solid voxel with `w>0` has its cell **directly below**
  (same `u,v`, height `w-1`) occupied.
- **Support grounding (<=45 deg):** every support voxel with `w>0` has at least one
  occupied cell in the 3x3 block below it (`u+-1, v+-1, w-1`). Support at `w=0` rests on
  the plate.

## Objective (minimise)
`F = |S| + LAM * U`, where `|S|` = number of support voxels and `U` = number of them that
are **not extractable**. A support voxel is **extractable** iff some straight axis ray in
the build frame — `+u, -u, +v, -v` (lateral) or `+w` (up) — leaves the bounding box while
passing through **no solid** cell (other support along the ray is fine; `-w` is blocked by
the plate). Minimising material and guaranteeing extraction pull against each other, and the
link between them is the orientation.

## Scoring
Let `B` be the cost of the checker's reference: orientation `o=0`, a straight vertical
support column beneath every overhang down to the first solid/plate, scored by the same
`F`. With minimisation normalisation:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Reproducing the reference gives `Ratio = 0.1`; a solution `10x` cheaper caps at `1.0`.

## Constraints
- Grids satisfy `X,Y,Z <= 40`; at least one overhang exists so `B > 0`.
- `LAM >= 1`. Time limit 5s, memory 512m.

## Example
A hollow box with a single floating voxel inside and one narrow slot cut in a wall. Dropping
a straight column under the floating voxel (obvious, least material) leaves every column
voxel walled in — none can see out, so `U` is large. Sloping the column sideways to line up
with the slot, or first turning the box so the slot faces outward instead of down, keeps
almost the same material but makes the whole column extractable — a much smaller `F`.
```
Reference F = 60  =>  a solution with F = 20 scores  Ratio = min(1000, 100*60/20)/1000 = 0.3
```
