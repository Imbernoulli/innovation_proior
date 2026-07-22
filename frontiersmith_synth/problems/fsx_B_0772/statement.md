# Crossed Stacks: Slicing a Plywood Sculpture at the Pin-Sheet Trade

A sculptor builds pieces out of **crossed plywood stacks**: a solid shape is sliced along
one direction into thin rectangular cross-sections ("slabs"), each slab is cut from a
stock plywood sheet, and the slabs are re-stacked to rebuild the exact solid. Different
parts of a sculpture may be sliced along different directions -- but wherever two
differently-sliced parts meet, the grain no longer lines up across the seam, and the
sculptor must drive an **alignment pin** through every unit of that seam's area to hold
it together. Pins are a scarce, budgeted resource.

## Model

The sculpture is a solid built from `NB` axis-aligned rectangular blocks on an integer
grid `[0,X) x [0,Y) x [0,Z)` (their union is the solid). Choose a set of `R` axis-aligned
**regions** (boxes) that exactly partition the solid -- every solid cell in exactly one
region, no region touching empty space -- and assign each region a **slicing axis**
`a in {0,1,2}` (X, Y or Z).

Slicing a region of extents `(dx,dy,dz)` along axis `a` produces one slab per layer along
`a`, each a rectangle with the region's other two extents. A slab of size `p x q` fits a
stock sheet `SW x SH` iff `(p<=SW and q<=SH)` or `(q<=SW and p<=SH)` (rotation is free).
**A slab fitting neither orientation makes the whole submission infeasible** -- that
region's axis is physically impossible to mill from your stock.

Wherever two regions share a face of positive area and use **different** axes, you spend
one alignment pin per unit area of that shared face. Total pins spent may not exceed the
budget `P` given in the input.

## Input (stdin)
```
X Y Z
NB
x0 y0 z0 x1 y1 z1        # NB lines: half-open blocks [x0,x1)x[y0,y1)x[z0,z1); union = solid
SW SH
P
```

## Output (stdout)
```
R
x0 y0 z0 x1 y1 z1 axis   # R lines, one region each; axis in {0,1,2}
```

## Feasibility
- The `R` regions are axis-aligned boxes strictly inside the grid, pairwise cell-disjoint,
  and their union is exactly the solid (no gaps, no cells outside the solid).
- Every region's slab fits some stock sheet under its assigned axis (see above).
- Total alignment pins spent (summed over every differently-axed touching region pair) is
  at most `P`.
Any violation scores `0`.

## Scoring
Slabs of identical size (after choosing the sheet-fitting orientation) can be nested
together across regions: a size class with total slab count `n` and per-sheet capacity
`k = floor(SW/w) * floor(SH/h)` needs `ceil(n/k)` sheets. Your objective `F` is the sum of
`ceil(n/k)` over all size classes actually produced. Let `B` be the sheet count of the
simplest always-feasible construction (each input block kept as its own region, sliced
along the first axis in `X,Y,Z` order whose cross-section fits a sheet). Since this is a
minimization,
```
score = min(1000, 100 * B / F) / 1000
```
Reproducing the naive per-block construction scores `0.10`; using 10x fewer sheets caps
at `1.0`.

## What makes this hard
A single axis for the whole sculpture never pays a pin, but a thin feature running
"across" that axis may not fit any sheet -- a sculpture with protrusions in two different
directions has **no feasible single axis**, full stop. Optimizing each block's axis alone
always fits, but ignores neighbours: its tie-breaks often pick an axis that needlessly
disagrees with a cheap-to-match neighbour, or misses that pooling several blocks'
identically-shaped slabs onto shared sheets beats slicing each on its own. The right axis
per block, its sheet cost, and whether matching a neighbour is worth the pins it costs,
all depend on the exact geometry in the input -- no shape-independent rule works.

## Example
A `2x2x3` block with one `1x1x4` arm sticking out along Y needs `SW=SH=3` sheets. Slicing
everything along Z (matching nothing) needs 5 sheets. Slicing the block along its own
best axis and the arm along Y independently (paying 1 pin for the mismatched seam, area
1) needs only 4 sheets -- worth it whenever `P>=1`.

## Constraints
`X,Y,Z <= 45`, `2 <= NB <= 6`, `R <= 30`, `1 <= SW,SH <= 20`, `0 <= P <= 50`. Time limit
5s, memory 512MB. All arithmetic is exact integers -- fully deterministic.
