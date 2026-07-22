# One Artery Feeds Every Organ: Trunks That Split Late

## Problem
A single arterial **source** must perfuse a set of **organs** (sinks) in the plane. You
design the vascular tree: a network of tubes, rooted at the source, that reaches every
organ. You may introduce **junction** (Steiner) points anywhere so the tree can branch in
empty space. Each organ has a positive **demand**; by conservation, every tube carries the
total demand of the organs downstream of it (its **flow**).

By **Murray's law** a tube carrying flow `f` has radius `~ f^(1/3)`, hence cross-section
`~ f^(2/3)`, so its material cost per unit length is *concave* in flow — one thick shared
trunk is cheaper than many thin parallel vessels carrying the same total. Against that pulls
a *linear* **delivery** cost that rewards short, direct paths to each organ. Rigid
**obstacles** (rectangles) sit in the tissue; no tube may pass through an obstacle's
interior (touching the boundary is allowed).

## Input (stdin)
```
K M Wm Wd
sx sy
<K lines>  ox oy d      # organ: coordinates and demand d > 0
<M lines>  x0 y0 x1 y1  # obstacle: axis-aligned rectangle, x0<x1, y0<y1
```
`K` organs, `M` obstacles, material weight `Wm`, delivery weight `Wd`, source `(sx,sy)`.

## Output (stdout)
Describe the tree over `1 + K + P` nodes: node `0` is the source, nodes `1..K` are the
organs (in input order), nodes `K+1..K+P` are junctions you add.
```
P                      # number of junctions you add (P >= 0)
<P lines>  jx jy       # junction coordinates
<K+P lines> parent_i   # for node i = 1,2,...,K+P: the index of its parent (0..K+P)
```
Every node's parent chain must reach the source `0` (a rooted tree; no cycles). Junction
coordinates must be finite.

## Feasibility
An output is rejected (**Ratio 0.0**) unless all hold: `0 <= P` (and not absurdly large);
every parent is a valid node index, `parent_i != i`, and following parents from any node
reaches `0` without cycling; every tube — the segment from a node to its parent — avoids the
strict interior of **every** obstacle. Wiring each organ straight to the source is always
feasible (no obstacle blocks a source→organ line).

## Objective (minimize)
For an edge `e` let `len(e)` be its Euclidean length and `flow(e)` the summed demand of the
organs in the subtree it feeds. Minimize
```
F  =  sum over edges e of  len(e) * ( Wm * flow(e)^(2/3)  +  Wd * flow(e) ).
```
The first term is Murray tube volume (concave — aggregation pays); the second is
demand-weighted delivery length (linear — directness pays). Junction placement and which
flows to merge are yours to choose.

## Scoring
The checker builds the **star** baseline `B` (every organ wired straight to the source) and
reports, for minimization,
```
Ratio = min(1000, 100 * B / max(1e-9, F)) / 1000.
```
Reproducing the star gives `Ratio = 0.1`; halving the cost gives `0.2`; a 10x reduction
caps at `1.0`. Higher is better.

## Constraints
- `25 <= K <= 55`, `0 <= M <= 2`, `Wm = 1`, `Wd = 0.03`.
- Coordinates fit in `[-350, 350]`; demands are small positive numbers.
- `P <= 6*K + 4`. Time limit 3s, memory 512m.

## Example
Two organs of demand 1 sit close together, far up the y-axis, with the source at the origin.
The star runs two long tubes, cost `~ 2 * L * (Wm*1 + Wd*1)`. Instead run one trunk from the
source to a junction just below the pair, then two short tubes: the trunk carries flow `2`,
costing `~ L * (Wm*2^(2/3) + Wd*2)` over almost the whole length, and `2^(2/3) ≈ 1.587 < 2`,
so the shared haul is cheaper than two separate ones — the fan-out happens late. Discovering
*where* to split, and which organs to merge, against the delivery pull and the obstacles, is
the open problem.
