# Context: AHC039 "Purse Seine Fishing" — grid-cell greedy region growing

## Research question

AtCoder AHC039: place one axis-aligned **rectilinear simple polygon** (the net) in the sea
`[0, 10^5] × [0, 10^5]` to maximize `max(0, a − b + 1)`, where `a` = mackerel caught and
`b` = sardine caught (fish inside or on the boundary count). Constraints: vertex count
`4 ≤ m ≤ 1000`; axis-parallel edges; simple polygon; perimeter `≤ 4 × 10^5`. There are
`N = 5000` mackerel and `N = 5000` sardine, drawn from overlapping clustered shoals, so the species
interleave and the right net is an irregular rectilinear region — not a box.

## Why grid cells

A single rectangle (the natural baseline) is convex and cannot bend its boundary around interior
sardine. The problem allows up to a thousand vertices, so we want a net whose boundary follows the
*shape* of where the mackerel are. The most direct way to get an arbitrary rectilinear region is to
build it from grid cells: bucket the sea into a `G × G` grid, give each cell weight
`(#mackerel − #sardine)`, and let the net be a connected, hole-free subset of cells. The outer
boundary of such a subset is automatically a simple rectilinear polygon, and `a − b` is the sum of
the chosen cells' weights — turning "design a rectilinear net" into "select a good connected cell
region."

## The greedy and its hazards

Grow from the densest cell, at each step adding the highest-weight admissible frontier cell. Two
invariants must hold under each add: the running boundary-edge count must keep the traced perimeter
`≤ 4 × 10^5`, and the region must stay hole-free (checked by flood-filling the outside from the grid
border). A purely-positive greedy freezes early on the sardine collar around a good region, so we
allow bridging through negative cells while tracking and restoring the best total weight seen.
Because one resolution is never right for every layout, we sweep a few `G` values and keep the best;
and we include the best prefix-sum rectangle as a candidate so the method never scores below the
rectangle baseline.

## Evaluation

Frozen local harness faithful to AHC039 (generator from 2D-Gaussian shoals; exact integer
point-in-rectilinear-polygon evaluator with full validity checks; boundary ⇒ inside). Five seeded
instances (seeds 1–5), 5000 mackerel + 5000 sardine each; reported metric is the raw mean objective
`a − b + 1`. AtCoder performance frontier (relative scale, not the raw objective): ALE-Agent `2880`
(5th) → ShinkaEvolve `3140` (2nd).

## Limitation this method exposes

The greedy is a single forward pass with no reversibility: it cannot remove a cell it later regrets,
it spends its perimeter budget on whatever ragged boundary it stumbles into, and it is at the mercy
of a static grid resolution. Those three failures are all failures of irreversibility — the opening
for replacing greedy growth with reversible local *search* (simulated annealing).
