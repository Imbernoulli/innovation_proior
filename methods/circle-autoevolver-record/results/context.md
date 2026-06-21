# Context: sum-of-radii circle packing in the unit square (n = 26) — the frontier

## Research question

Place `26` circles inside the unit square `[0,1]²`, pairwise non-overlapping, and maximize the
**sum of their radii** `Σ rᵢ`. The constructor emits one packing — centers `(xᵢ, yᵢ)` and radii
`rᵢ ≥ 0` — scored by `Σ rᵢ` alone. The radii are **free and unequal**: a good packing mixes a few
large circles with many small gap-fillers, and the optimum is irregular with no symmetry.

## Constraints (feasibility)

- Inside the square: `rᵢ ≤ xᵢ ≤ 1 − rᵢ`, `rᵢ ≤ yᵢ ≤ 1 − rᵢ`.
- Pairwise disjoint: `√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ` for `i ≠ j`.
- `rᵢ ≥ 0`. Accepted when no constraint is violated beyond a small absolute tolerance. The
  AutoEvolver/OpenEvolve harness uses `atol=1e-6`; ShinkaEvolve uses `1e-7`.

Nonconvex QCQP. For **fixed centers, the optimal radii are an LP**; the nonconvex part is the
placement of the centers, in a landscape of many basins of differing quality.

## Prior art and the frontier

Published frontier for `n = 26`: AlphaEvolve `2.63586276` (arXiv:2506.13131, improving Friedman-2012
`~2.634`), ShinkaEvolve `2.635983283` (arXiv:2509.19349), ThetaEvolve `2.63598308`, and AutoEvolver
/ Claude Code `2.635988438567568` (github.com/tengxiaoliu/autoevolver,
tengxiaoliu.github.io/autoevolver), found with `~16.6 h` autonomous compute. These program-evolution
/ agentic systems converged on the same hybrid pipeline — structured initialization (golden-angle
spiral + corner/edge seeding), joint SLSQP refinement of centers *and* radii together, and iterated
perturbation chains / simulated annealing on the incumbent. The frontier is a band separated by only
parts in the sixth decimal place; each part corresponds to a long sequence of mostly-lateral
perturb-and-re-SLSQP moves through near-degenerate basins.

A bounded (~9 min) run of that exact pipeline reaches the frontier *neighborhood* at `≈ 2.6275`,
about `0.0085` below the best-published value.

## This method's role

This is the **record-reaching rung** of the ladder. The ladder's own pipeline — structured
restarts, joint SLSQP, iterated perturbation chains — is already the frontier construction. The
configurations at the very top of the published band were each established by long autonomous search,
and the AutoEvolver value `2.635988438567568` is the highest reported for `n = 26`. This rung
operates at that top of the band, within the harness that scores packings. The harness tolerance is
`atol=1e-6`: a packing is accepted when no constraint is violated by more than `1e-6`. Frontier-band
configurations press contacts close to that tolerance, so feasibility verdicts can differ between
`atol=1e-6` and the stricter `1e-7`.
