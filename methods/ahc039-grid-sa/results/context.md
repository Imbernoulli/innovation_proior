# Context: AHC039 "Purse Seine Fishing" — simulated annealing on a grid-cell net

## Research question

AtCoder AHC039: design one axis-aligned **rectilinear simple polygon** (the net) in
`[0, 10^5] × [0, 10^5]` maximizing `max(0, a − b + 1)` — mackerel caught minus sardine caught, plus
one — under `4 ≤ m ≤ 1000` vertices, axis-parallel edges, simplicity, and perimeter `≤ 4 × 10^5`.
`N = 5000` mackerel and `N = 5000` sardine come from overlapping clustered shoals, so the net must
be an irregular rectilinear region that hugs mackerel-dense pockets while routing around sardine,
all under a binding perimeter budget.

## Why local search

A grid-cell greedy region (the prior method) bends its boundary around sardine but is a one-shot,
irreversible forward pass: it cannot undo an early bad inclusion, wastes perimeter on ragged
boundaries, and depends on a single grid resolution. The fix is to keep the representation — a
connected, hole-free set of grid cells whose outer boundary is a simple rectilinear polygon — and
search it with moves that can both **add and remove** cells, accepting downhill steps to escape the
greedy's traps. This is the established strong baseline for AHC039: ALE-Agent ran simulated
annealing on the net with a spatial index and incremental scoring, reaching AtCoder performance
`2880` (5th place).

## The machinery that makes SA affordable

Precompute per-cell mackerel and sardine counts, so a single-cell add/remove flip changes `a − b`
in O(1) (incremental scoring — no recount during the search). Keep a running boundary-edge count for
an O(1) perimeter check. Preserve topology with the constant-time digital-topology **simple-point**
test on the 3×3 ring around a flipped cell, plus a 2×2 diagonal-pinch guard so the boundary stays a
single simple cycle. **Warm-start** the region to the best perimeter-constrained rectangle (a
prefix-sum sweep) — because perimeter is the binding constraint and a rectangle is the most
perimeter-efficient seed — so SA spends its whole budget carving notches rather than inflating a
blob. Anneal with Metropolis acceptance on `Δ(a−b)` and geometric cooling; keep the best region.

## Evaluation

Frozen local harness faithful to AHC039 (clustered-shoal generator; exact integer
point-in-rectilinear-polygon evaluator with full validity checks; boundary ⇒ inside). Five seeded
instances (seeds 1–5), 5000 + 5000 fish; reported metric is the raw mean objective `a − b + 1`,
~5 s per instance. AtCoder performance frontier (relative scale): ALE-Agent `2880` (5th) →
ShinkaEvolve `3140` (2nd).

## Limitation this method exposes

The single-cell flip is *local* and the perimeter budget is *global* and binding: once the boundary
is near its length limit, the useful moves are coordinated "shave here, extend there" trades that a
blind random flip rarely proposes — most flips touch already-correct boundary. That undirectedness
(and the per-candidate revalidation cost) are exactly the two levers ShinkaEvolve sharpened to lift
the same SA to `3140`.
