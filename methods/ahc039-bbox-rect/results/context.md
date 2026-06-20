# Context: AHC039 "Purse Seine Fishing" — the bounding-box baseline

## Research question

AtCoder AHC039 asks for a single **net**: a closed, axis-aligned *rectilinear* simple polygon
placed in the sea `[0, 10^5] × [0, 10^5]`. In the sea are `N = 5000` **mackerel** (targets) and
`N = 5000` **sardine** (penalties), each at an integer coordinate, drawn from overlapping clustered
shoals. Every fish inside the net or on its boundary is caught, and the case is scored

```
score = max(0, a − b + 1),
```

with `a` = mackerel inside, `b` = sardine inside. The net must satisfy: vertex count `4 ≤ m ≤ 1000`;
every edge axis-parallel; the polygon simple (no self-intersection); perimeter (total edge length)
`≤ 4 × 10^5`. The design target is the geometry of the net; nothing else is tunable.

## Why a rectangle first

Before any clever region-shaping, we want the simplest *guaranteed legal* net and a concrete
baseline objective. A single axis-aligned rectangle is always legal: four vertices (inside the `m`
bounds), four axis-parallel edges, trivially simple, and its perimeter `2(w+h)` is controllable.
The only question is which rectangle maximizes `a − b`.

## The structure that makes it tractable

Stamp weight `+1` at every mackerel and `−1` at every sardine. The value `a − b` of any
axis-aligned box equals the total signed weight it covers, so the best rectangle is the
**maximum-weight axis-aligned rectangle** on a signed point set — a classic problem solvable by
coordinate discretization plus 2D prefix sums: bucket fish into a coarse grid of candidate cut
lines, build a prefix-sum table of per-cell `(#mackerel − #sardine)`, and read any grid-aligned
box's value in O(1), sweeping all boxes under the perimeter budget.

## Evaluation

A frozen local harness faithful to AHC039: a generator places 5000 mackerel + 5000 sardine from
random 2D-Gaussian shoals under a fixed seed; an exact evaluator validates the net (vertex bounds,
axis-parallel edges, simplicity, perimeter, integer coords in range) and counts fish inside via an
exact integer point-in-rectilinear-polygon ray-cast (boundary ⇒ inside), returning `max(0, a−b+1)`.
Five seeded instances (seeds 1–5); the reported metric is the raw mean objective. A C++ and an
independent Python evaluator are cross-checked to agree exactly. The AtCoder performance frontier
for this task is ALE-Agent `2880` (5th) → ShinkaEvolve `3140` (2nd); those are on AtCoder's relative
scale, not the raw objective reported here.

## Limitation this baseline exposes

A rectangle is convex and single-bodied: it cannot carve a sardine pocket out of a mackerel shoal
nor bridge two separated mackerel pockets without swallowing the sardine-rich gap between them.
That convex-body limitation — not any tuning — is what motivates moving to an arbitrary rectilinear
region built from grid cells.
