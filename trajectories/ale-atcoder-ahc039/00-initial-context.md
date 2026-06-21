## Research question

AtCoder AHC039 — "Purse Seine Fishing." The sea is the square `[0, 10^5] × [0, 10^5]`. In it
swim `N = 5000` **mackerel** (targets) and `N = 5000` **sardine** (penalties), each at an integer
coordinate. The task is to design one **net**: a closed, axis-aligned rectilinear simple polygon.
Every fish strictly inside the net — or lying on its boundary — is caught. The score is

```
score(case) = max(0, a − b + 1),
```

where `a` is the number of mackerel inside the net and `b` the number of sardine inside. The net
must satisfy four hard constraints: its vertex count `m` obeys `4 ≤ m ≤ 1000`; every edge is
parallel to the x- or y-axis; the polygon is simple (no self-intersection); and the total edge
length (perimeter) is at most `4 × 10^5`. The objective is the raw value `a − b + 1`; there is no
partial credit beyond which fish the polygon happens to enclose.

The tension is entirely geometric. A bigger net catches more mackerel but also more sardine and
spends more of the scarce perimeter budget; a tighter net excludes sardine but may miss outlying
mackerel and must pay for every staircase notch it cuts. Because mackerel and sardine are drawn
from overlapping clustered shoals, the two species interleave, and the right net is an irregular
rectilinear region that hugs mackerel-dense pockets while routing its boundary around sardine
clusters — all under a perimeter ceiling that makes boundary length a hard currency.

## Prior art / Background / Baselines

The published contest frontier provides two reference points:

| Reference point | AtCoder performance (relative scale) |
|---|---|
| ALE-Agent solution (simulated annealing + kd-tree) | **2880** (5th place) |
| Best published refinement | **3140** (2nd place) |

Local raw-objective means live on an absolute scale and are not directly comparable to AtCoder's
relative performance scale; the contest numbers give only the competitive context.

- **Single axis-aligned bounding box.** Search all axis-aligned rectangles via 2D prefix sums to
  maximize `a − b`. *Gap:* a single convex rectangle cannot carve sardine out of a mackerel shoal
  nor reach two separated mackerel pockets without swallowing the sardine-rich gap between them.
- **Grid-cell greedy region growing.** Bucket the sea into a coarse grid, score each cell by
  `(#mackerel − #sardine)`, and greedily grow a connected, hole-free cell region under the
  perimeter budget, then trace its rectilinear boundary. *Gap:* coarse grid quantization freezes
  the boundary at cell resolution, and a local frontier greedy easily gets trapped by the perimeter
  ceiling without any way to undo an early bad inclusion.
- **Simulated annealing on the polygon (ALE-Agent).** Run SA over the net, moving the boundary and
  adding/removing staircase steps, with a spatial index (kd-tree) and incremental scoring so each
  candidate is cheap. This is the established strong method and sits at 2880 (5th). *Gap:* the
  search plateaus below the best known scores; many accepted boundary moves reshape the net without
  improving the objective, so long runs yield only marginal gains.

## Fixed substrate / Code framework

The harness is a faithful local reproduction of AHC039. A **generator** places 5000 mackerel and
5000 sardine, each species drawn from a random mixture of 2D-Gaussian shoals (clustered
placements, clipped to the grid), under a fixed seed. An **evaluator** reads a net and validates
`4 ≤ m ≤ 1000`, integer vertices in `[0, 10^5]`, every edge axis-parallel, polygon simplicity, and
perimeter `≤ 4 × 10^5`; then counts mackerel and sardine inside or on the boundary via an exact
integer point-in-rectilinear-polygon ray-cast (boundary ⇒ inside) and returns `max(0, a − b + 1)`.
The generator, the evaluator, the constraints, and the seed set are frozen. A C++ evaluator and an
independent Python evaluator are cross-checked to agree exactly.

## Editable interface

Exactly one thing is editable: the **solver** that reads an instance (the `N`, the mackerel, the
sardine) on stdin and writes a net (`m`, then `m` vertices) on stdout. Every rung on the ladder is
a different solver. The contract is fixed: the output must be a legal net by the rules above; the
score is whatever the frozen evaluator returns. Nothing about the generator, the scoring, or the
constraints is learnable — the only lever is the geometry of the net the solver emits.

## Evaluation settings

Five seeded instances (seeds 1–5), each 5000 mackerel + 5000 sardine from clustered shoals. We
report the raw mean objective `a − b + 1` over the five seeds for each rung, computed by the frozen
exact evaluator. Each rung is run on the *same* five instances so the numbers are directly
comparable. The reported metric is the literal objective the evaluator computes; the AtCoder
performance numbers are cited only as competitive context.
