# Context: AHC039 "Purse Seine Fishing" — ShinkaEvolve refinements (cached validation + targeted edge move)

## Research question

AtCoder AHC039: design one axis-aligned **rectilinear simple polygon** (the net) in
`[0, 10^5] × [0, 10^5]` maximizing `max(0, a − b + 1)` — mackerel caught minus sardine caught, plus
one — under `4 ≤ m ≤ 1000` vertices, axis-parallel edges, simplicity, and perimeter `≤ 4 × 10^5`,
with `N = 5000` mackerel and `N = 5000` sardine from overlapping clustered shoals. This is the
endpoint method: the genuine best-known refinements layered on top of the grid-cell simulated
annealer.

## The frontier this reproduces

ALE-Bench (arXiv:2506.09050) reports ALE-Agent solving AHC039 with simulated annealing over the net
plus a kd-tree spatial index, reaching AtCoder performance `2880` (5th place). ShinkaEvolve
(arXiv:2509.19349), an LLM-driven program-evolution system, was seeded with that solution and
evolved it to `3140` (2nd place) with exactly two changes:

1. **Caching the validation process.** The kd-tree was augmented to cache subtree statistics —
   bounding boxes and fish counts — at each node, so candidate validation and scoring became
   incremental rather than re-walking the structure.
2. **A novel "targeted edge move."** A neighborhood operator that heuristically identifies a
   misclassified fish (e.g. a mackerel outside the polygon) and greedily moves the nearest edge to
   correct its state. Together these "strengthened the directionality of the search."

## How they port to a grid-cell net

On top of the warm-started grid SA (region = connected hole-free cell set; O(1) incremental `a−b`
from per-cell fish counts; simple-point + pinch topology checks; geometric cooling), the two levers
become: (1) a **per-cell boundary-flag cache** refreshed only on the 3×3 neighbourhood of each
accepted flip — so proposals and validity read a cache instead of rescanning the grid, the analogue
of caching subtree statistics; and (2) a **targeted edge move** — sample a boundary cell and
greedily ADD a mackerel-rich outside neighbour (capture a missed mackerel) or REMOVE a sardine-heavy
boundary cell (release a caught sardine), aiming proposals at fish the net misclassifies. Directed
proposals are mixed with uniform random flips and still pass through Metropolis acceptance and every
validity gate.

## Evaluation

Frozen local harness faithful to AHC039 (clustered-shoal generator; exact integer
point-in-rectilinear-polygon evaluator with full validity checks; boundary ⇒ inside). Five seeded
instances (seeds 1–5), 5000 + 5000 fish; reported metric is the raw mean objective `a − b + 1`,
~5 s per instance, reproducible across runs. Scale note: the raw objective is *not* on AtCoder's
relative performance scale; `2880 → 3140` are cited as the frontier context this method's ideas
reproduce, not as a scale the raw mean lives on.

## What this is and is not

This is the SA made to search in the right direction with an affordable cache — the same character
as the benchmark's `2880 → 3140`, a modest but real and reproducible gain, not a new algorithm.
Further gains on the real task come from a different kind of search (continuous-polygon
representations, multi-net restarts), which is outside this local-move annealer.
