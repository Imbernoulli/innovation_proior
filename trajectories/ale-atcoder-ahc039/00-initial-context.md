## Research question

AtCoder AHC039 — "Purse Seine Fishing." The sea is the square `[0, 10^5] × [0, 10^5]`. In it
swim `N = 5000` **mackerel** (targets) and `N = 5000` **sardine** (penalties), each at an integer
coordinate. The single thing being designed is one **net**: a closed, axis-aligned *rectilinear*
simple polygon. Every fish strictly inside the net — or lying on its boundary — is caught. The
catch is scored by

```
score(case) = max(0, a − b + 1),
```

where `a` is the number of mackerel inside the net and `b` the number of sardine inside. The net
must obey four hard constraints: its vertex count `m` satisfies `4 ≤ m ≤ 1000`; every edge is
parallel to the x- or y-axis; the polygon is simple (no self-intersection); and the total edge
length (perimeter) is at most `4 × 10^5`. A constructor that emits a net is read against the raw
objective `a − b + 1` alone — there is no partial credit beyond which fish the polygon happens to
enclose.

The tension is entirely geometric. A bigger net catches more mackerel but also more sardine and
spends more of the scarce perimeter budget; a tighter net excludes sardine but may miss outlying
mackerel and must pay for every staircase notch it cuts. Because mackerel and sardine are drawn
from overlapping clustered shoals, the two species interleave, and the right net is an irregular
rectilinear region that hugs mackerel-dense pockets while routing its boundary around sardine
clusters — all under a perimeter ceiling that makes boundary length a hard currency.

## How the score is defined

The objective is `max(0, a − b + 1)` per case, summed across cases on AtCoder. The `+1` keeps an
empty-but-legal net (catching nothing) at score `1` rather than `0`, and the `max(0, ·)` floors a
net that catches more sardine than mackerel. We report the **raw mean objective** `a − b + 1`
averaged over a fixed set of seeded local instances. This raw number is *not* on AtCoder's
relative scale — AtCoder maps each case to a performance score relative to the field — so the
honest yardsticks to keep in view are the published benchmark performance numbers, not the raw
objective:

| Reference point | AtCoder performance (relative scale) |
|---|---|
| ALE-Agent solution (simulated annealing + kd-tree) | **2880** (5th place) |
| **ShinkaEvolve refinement** (cached validation + targeted edge move) | **3140** (2nd place) |

These two numbers — ALE-Agent's `2880` and ShinkaEvolve's `2880 → 3140` — are the frontier this
ladder is built against (ALE-Bench, arXiv:2506.09050; ShinkaEvolve, arXiv:2509.19349). Our local
raw-objective means live on a different (absolute) scale; the ladder's job is to climb the raw
objective monotonically while reproducing the *ideas* that took ALE-Agent to ShinkaEvolve.

## Prior art before the first rung

- **Single axis-aligned bounding box.** The trivial legal net is one rectangle. Choosing it to
  maximize `a − b` is a 2D-prefix-sum search over coordinate-compressed rectangles. *Gap:* a
  rectangle cannot carve sardine out of a mackerel shoal nor reach two separated mackerel pockets
  without swallowing the sardine-rich gap between them — it is a single convex box against an
  irregular, multi-modal target.
- **Grid-cell greedy region growing.** Bucket the sea into a coarse grid, score each cell by
  `(#mackerel − #sardine)`, and greedily grow a connected, hole-free cell region under the
  perimeter budget, then trace its rectilinear boundary. *Gap:* a purely local frontier greedy is
  fragile — it gets trapped by the perimeter ceiling, and a coarse static grid both quantizes the
  boundary and cannot undo an early bad inclusion.
- **Simulated annealing on the polygon (ALE-Agent).** The benchmark baseline runs SA over the
  net, moving the boundary and adding/removing staircase steps, with a spatial index (kd-tree) and
  incremental scoring so each candidate is cheap. This is the established strong method and reached
  `2880` (5th). *Gap:* the per-candidate validation and the neighborhood operators are the
  bottleneck — ShinkaEvolve found that caching subtree statistics and adding a *directed* operator
  lifts the same SA to `3140` (2nd).
- **ShinkaEvolve refinements (endpoint).** An LLM-driven program-evolution system that, seeded
  with ALE-Agent's SA solution, evolved exactly two changes: (1) caching the validation process —
  augmenting the kd-tree to cache subtree statistics (bounding boxes and fish counts) at each node;
  and (2) a novel "targeted edge move" — heuristically identify a misclassified fish (e.g. a
  mackerel outside the net) and greedily move the nearest edge to correct its state. Together these
  "strengthened the directionality of the search," lifting `2880 → 3140`.

## The fixed substrate

The harness is a faithful local reproduction of AHC039. A **generator** places `5000` mackerel
and `5000` sardine, each species drawn from a random mixture of 2D-Gaussian shoals (clustered
placements, clipped to the grid), under a fixed seed. An **evaluator** reads a net and: validates
`4 ≤ m ≤ 1000`, integer vertices in `[0, 10^5]`, every edge axis-parallel, the polygon simple (no
two non-adjacent edges touch), and perimeter `≤ 4 × 10^5`; then counts mackerel and sardine inside
or on the boundary via an exact integer point-in-rectilinear-polygon ray-cast (boundary ⇒ inside)
and returns `max(0, a − b + 1)`. The generator, the evaluator, the constraints, and the seed set
are frozen. A C++ evaluator and an independent Python evaluator are cross-checked to agree exactly.

## The editable interface

Exactly one thing is editable: the **solver** that reads an instance (the `N`, the mackerel, the
sardine) on stdin and writes a net (`m`, then `m` vertices) on stdout. Every rung on the ladder is
a different solver. The contract is fixed: the output must be a legal net by the rules above; the
score is whatever the frozen evaluator returns. Nothing about the generator, the scoring, or the
constraints is learnable — the only lever is the geometry of the net the solver emits.

## Evaluation settings

Five seeded instances (seeds 1–5), each `5000` mackerel + `5000` sardine from clustered shoals. We
report the **raw mean objective** `a − b + 1` over the five seeds for each rung, computed by the
frozen exact evaluator. Each rung is run on the *same* five instances so the numbers are directly
comparable. The reported metric is honest and absolute (it is the literal objective the evaluator
computes); the AtCoder performance numbers `2880 → 3140` are cited as the frontier context the
endpoint reproduces, not as a scale our raw means live on.
