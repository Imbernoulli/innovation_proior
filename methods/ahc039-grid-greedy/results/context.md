# Context: AHC039 "Purse Seine Fishing" — designing a rectilinear net

## Research question

AtCoder AHC039: place one axis-aligned **rectilinear simple polygon** (the net) in the sea
`[0, 10^5] × [0, 10^5]` to maximize `max(0, a − b + 1)`, where `a` = mackerel caught and
`b` = sardine caught (fish inside or on the boundary count). Constraints: vertex count
`4 ≤ m ≤ 1000`; axis-parallel edges; simple polygon; perimeter `≤ 4 × 10^5`. There are
`N = 5000` mackerel and `N = 5000` sardine, drawn from overlapping clustered shoals, so the species
interleave across the plane.

The problem allows up to a thousand vertices and any rectilinear shape; the question is how to
design and search over such a net so it captures as much of the mackerel-minus-sardine surplus as
the layout permits.

## Baseline

A single axis-aligned **rectangle** is the natural starting point: a four-vertex net chosen by a 2D
prefix-sum sweep over the species counts. Bucket the sea into a grid, give each cell weight
`(#mackerel − #sardine)`, build the prefix sums of those weights, and over all axis-aligned cell
rectangles whose perimeter is `≤ 4 × 10^5` pick the one of maximum summed weight. The rectangle uses
four of the thousand available vertices and its boundary is a box.

## A grid representation of arbitrary rectilinear nets

A direct way to express an arbitrary rectilinear region is to build it from grid cells: bucket the
sea into a `G × G` grid, weight each cell `(#mackerel − #sardine)`, and let the net be a connected,
hole-free subset of cells. The outer boundary of such a subset is a simple rectilinear polygon (the
unit edges where an inside cell meets an outside cell or the grid border), and `a − b` is the sum of
the chosen cells' weights. This recasts "design a rectilinear net" as "select a connected,
hole-free cell region," a combinatorial selection problem. Validity of a candidate region is two
local-to-global conditions: the traced boundary's perimeter must stay `≤ 4 × 10^5` (adding a cell
changes the boundary-edge count by a known local amount — it removes shared edges with inside
neighbors and adds its exposed edges), and the region must stay hole-free (no empty pocket enclosed,
testable by flood-filling the outside from the grid border).

## Evaluation

Frozen local harness faithful to AHC039 (generator from 2D-Gaussian shoals; exact integer
point-in-rectilinear-polygon evaluator with full validity checks; boundary ⇒ inside). Five seeded
instances (seeds 1–5), 5000 mackerel + 5000 sardine each; reported metric is the raw mean objective
`a − b + 1`. AtCoder performance frontier (relative scale, not the raw objective): ALE-Agent `2880`
(5th) → ShinkaEvolve `3140` (2nd).
