# Context: AHC039 "Purse Seine Fishing" — refining the grid-cell simulated annealer

## Research question

AtCoder AHC039: design one axis-aligned **rectilinear simple polygon** (the net) in
`[0, 10^5] × [0, 10^5]` maximizing `max(0, a − b + 1)` — mackerel caught minus sardine caught, plus
one — under `4 ≤ m ≤ 1000` vertices, axis-parallel edges, simplicity, and perimeter `≤ 4 × 10^5`,
with `N = 5000` mackerel and `N = 5000` sardine drawn from overlapping clustered shoals. Starting
from a warm-started grid-cell simulated annealer that already searches the net's shape, the question
is how to push the search further within that same local-move annealer.

## The base annealer

The starting point is a simulated annealer over a `G × G` binary grid (here `G = 50`). The region —
the set of "inside" cells — is a connected, hole-free cell set whose outer boundary traces a single
rectilinear simple polygon. Per-cell mackerel and sardine counts `(Av, Bv)` give an O(1) incremental
update to `a − b` when a cell is toggled, and a running boundary-edge count gives an O(1) perimeter
estimate. A move toggles one cell in or out; the topology is kept valid with a simple-point /
crossing-number test plus a diagonal-pinch check so the boundary stays a single cycle. Metropolis
acceptance on the change in `a − b` with a geometric cooling schedule (`T0 = 8.0 → T1 = 0.05`) over a
~5 s budget; the best region seen is retained and traced to a polygon at the end. The region is
warm-started from the best perimeter-feasible axis-aligned rectangle (found by 2D prefix sums of the
per-cell weight `Av − Bv`), giving the anneal a strong starting basin from which it carves staircase
notches.

## The frontier this reproduces

ALE-Bench (arXiv:2506.09050) reports ALE-Agent solving AHC039 with simulated annealing over the net
plus a kd-tree spatial index, reaching AtCoder performance `2880` (5th place). ShinkaEvolve
(arXiv:2509.19349), an LLM-driven program-evolution system, was seeded with that solution and evolved
it to performance `3140` (2nd place), reported as a small number of targeted edits that
"strengthened the directionality of the search." That published `2880 → 3140` result is the frontier
context this work reproduces in the grid-cell representation rather than the original kd-tree one.

## Evaluation

Frozen local harness faithful to AHC039: clustered-shoal generator; exact integer
point-in-rectilinear-polygon evaluator with full validity checks (boundary ⇒ inside). Five seeded
instances (seeds 1–5), 5000 + 5000 fish; reported metric is the raw mean objective `a − b + 1`,
~5 s per instance, reproducible across runs. Scale note: the raw objective is *not* on AtCoder's
relative performance scale; the cited `2880` and `3140` are the published frontier numbers, not a
scale the raw mean lives on.
