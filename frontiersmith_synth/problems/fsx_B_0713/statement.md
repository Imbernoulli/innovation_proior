# Canopy Spacing: A Tree That Outgrows Its Own Shade

## Problem
You design one growth rule for a symmetric L-system tree of a fixed depth `D` (given in
the input) growing in a vertical plane. Starting from the origin heading straight up with
length `L0`, every branch node splits into two children whose headings diverge from the
parent's heading by `+theta*(1+skew)` and `-theta*(1-skew)` degrees, and whose length is
the parent's length times `r`. This produces the full binary skeleton (`2^D - 1`
segments). Only the `2^(D-1)` terminal tips (depth `D`) bear a leaf, a disk of area
`A0*taper`.

The sun visits the canopy along a fixed schedule of `K` parallel-ray directions (angles
`alpha_k` measured from straight down, positive = east, with weights `w_k` summing to 1).
For one ray direction, project every leaf disk onto the axis perpendicular to the ray.
Process leaves **nearest-to-sun first** (by their position along the ray). A leaf's
harvested area is the portion of its own projected interval **not already covered** by
the projected intervals of leaves nearer to the sun (a nearer leaf blocks light with its
whole footprint, whether or not that leaf itself is fully lit). Harvested light for that
ray is the sum of harvested areas over all tips; the total harvested light is the
weight-averaged sum over the `K` rays.

Biomass cost is `cost_len * (total length of all 2^D - 1 segments) + cost_leaf * (total
leaf area of all tips)`. The objective is `F = harvested_light - biomass_cost`
(maximize). Piling on more length or leaf area is only good until self-shading and
structural cost eat the gain — the win is spacing tips apart, not growing more of them.

## Input (stdin)
```
D
L0 A0
cost_len cost_leaf
K
alpha_1 w_1
...
alpha_K w_K
```
`D` (canopy depth, 3..6), `L0`/`A0` (root length / reference leaf area, positive floats),
`cost_len`/`cost_leaf` (positive biomass-cost coefficients), then `K` sun rays each with
an angle in degrees and a weight (weights sum to 1).

## Output (stdout)
Exactly four numbers on one line: `r theta taper skew` — the length ratio, branching
half-angle in degrees, leaf-taper multiplier, and left/right asymmetry.

## Feasibility
The output must be exactly 4 finite numbers, with:
- `0.3 <= r <= 0.95`
- `1.0 <= theta <= 80.0`
- `0.3 <= taper <= 1.0`
- `-0.5 <= skew <= 0.5`
Any violation (wrong token count, non-numeric, non-finite, or out of range) scores
`Ratio: 0.0`.

## Scoring
Let `B` be the objective `F` obtained by the checker's own fixed reference parameters
`(r, theta, taper, skew) = (0.45, 38.0, 0.6, 0.0)` on the same instance (always positive).
With maximization normalization:
```
sc = min(1000.0, 100.0 * max(0.0, F) / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the reference scores `Ratio = 0.1`; roughly 10x the reference value caps at
`1.0`.

## Constraints
- `3 <= D <= 6`, so between 4 and 32 terminal tips.
- `1 <= K <= 3` sun rays.
- Time limit 5s, memory 512m.

## Example
Suppose `D = 3`, `L0 = A0 = 1`, `cost_len = 0.03`, `cost_leaf = 0.02`, and a single sun
ray `alpha_1 = 0, w_1 = 1`. The reference parameters give some `B > 0` (computed by the
checker). An output of `0.8 45.0 0.9 0.0` grows a wider, longer-limbed tree; if its
harvested light comfortably outweighs its extra structural cost, `F` exceeds `B` and the
ratio rises above `0.1` — but pushing `r` and `taper` to their maximum values without
widening `theta` enough will pack tips too close together, lose most of the extra area to
self-shading, and can score **worse** than the reference despite using far more biomass.
