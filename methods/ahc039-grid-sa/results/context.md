# Context: AHC039 "Purse Seine Fishing" — improving on a grid-cell net

## Research question

AtCoder AHC039: design one axis-aligned **rectilinear simple polygon** (the net) in
`[0, 10^5] × [0, 10^5]` maximizing `max(0, a − b + 1)` — mackerel caught minus sardine caught, plus
one — under `4 ≤ m ≤ 1000` vertices, axis-parallel edges, simplicity, and perimeter `≤ 4 × 10^5`.
`N = 5000` mackerel and `N = 5000` sardine come from overlapping clustered shoals, so the net must
be an irregular rectilinear region that hugs mackerel-dense pockets while routing around sardine,
all under a binding perimeter budget. The question is how to shape such a region — given a starting
representation as a connected set of grid cells — to push the objective higher.

## Prior method

A grid-cell greedy region (the prior method) represents the net as a connected, hole-free set of
grid cells whose outer boundary is a simple rectilinear polygon, and bends its boundary around
sardine in a single forward pass: it walks the grid once, deciding for each candidate cell whether
to include it. The representation — a connected, hole-free cell set with a simple rectilinear outer
boundary — is the one carried forward here.

## The AHC039 baseline landscape

Published AHC039 entries operate on the same grid-cell net representation and report on a relative
performance scale. ALE-Agent reached AtCoder performance `2880` (5th place) on this problem;
ShinkaEvolve reached `3140` (2nd place). These are the reference points for the present setting.

## Evaluation

Frozen local harness faithful to AHC039 (clustered-shoal generator; exact integer
point-in-rectilinear-polygon evaluator with full validity checks; boundary ⇒ inside). Five seeded
instances (seeds 1–5), 5000 + 5000 fish; reported metric is the raw mean objective `a − b + 1`,
~5 s per instance. AtCoder performance frontier (relative scale): ALE-Agent `2880` (5th) →
ShinkaEvolve `3140` (2nd).
