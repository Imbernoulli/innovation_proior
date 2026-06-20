# Context: sum-of-radii circle packing in the unit square (n = 26) — reaching the record

## Research question

Place `26` circles inside the unit square `[0,1]²`, pairwise non-overlapping, and maximize the
**sum of their radii** `Σ rᵢ`. The constructor emits one packing — centers `(xᵢ, yᵢ)` and radii
`rᵢ ≥ 0` — scored by `Σ rᵢ` alone. The radii are **free and unequal**: a good packing mixes a few
large circles with many small gap-fillers, and the optimum is irregular with no symmetry.

## Constraints (feasibility)

- Inside the square: `rᵢ ≤ xᵢ ≤ 1 − rᵢ`, `rᵢ ≤ yᵢ ≤ 1 − rᵢ`.
- Pairwise disjoint: `√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ` for `i ≠ j`.
- `rᵢ ≥ 0`. Accepted when no constraint is violated beyond a small absolute tolerance. The
  AutoEvolver/OpenEvolve harness under which the record was established uses `atol=1e-6`;
  ShinkaEvolve uses `1e-7`.

Nonconvex QCQP. For **fixed centers, the optimal radii are an LP**; the hard, nonconvex part is
*where to put the centers*, in a landscape of many basins of differing quality.

## Prior art and the frontier

Published frontier for `n = 26`: AlphaEvolve `2.63586276` (arXiv:2506.13131, improving Friedman-2012
`~2.634`), ShinkaEvolve `2.635983283` (arXiv:2509.19349), ThetaEvolve `2.63598308`, and the **record
AutoEvolver / Claude Code `2.635988438567568`** (github.com/tengxiaoliu/autoevolver,
tengxiaoliu.github.io/autoevolver), found with `~16.6 h` autonomous compute. These program-evolution
/ agentic systems converged on the same hybrid pipeline — structured initialization (golden-angle
spiral + corner/edge seeding), joint SLSQP refinement of centers *and* radii together, and iterated
perturbation chains / simulated annealing on the incumbent. The frontier is a band separated by only
parts in the sixth decimal place; each part is bought with a long sequence of mostly-lateral
perturb-and-re-SLSQP moves through near-degenerate basins.

A bounded (~9 min) run of that exact pipeline reaches the frontier *neighborhood* at `≈ 2.6275`,
about `0.0085` below the record — the gap being search budget, not algorithm.

## This method's role

This is the **record-reaching rung**: rather than a new constructor, it reproduces AutoEvolver's
published best-known configuration verbatim and verifies it against the harness. The exact `26`
centers and `26` radii are loaded from `record_config.json` and checked for non-negative radii,
in-square containment, and pairwise non-overlap at the harness tolerance `atol=1e-6`, confirming
`Σ rᵢ = 2.635988438567568`. One honesty point: this configuration presses every contact to the edge
of the accepted tolerance (max constraint violation `≈ 8.81×10⁻⁷`, including a pairwise overlap of
`≈ 8.81×10⁻⁷`), so it is feasible at the harness `atol=1e-6` but **not** at the stricter `1e-7`. The
record is AutoEvolver's (Claude/Opus) published best-known, not a rediscovery by a bounded run.
