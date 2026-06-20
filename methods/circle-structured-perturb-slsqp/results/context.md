# Context: sum-of-radii circle packing in the unit square (n = 26)

## Research question

Place `26` circles inside the unit square `[0,1]²`, pairwise non-overlapping, and maximize the
**sum of their radii** `Σ rᵢ`. The constructor emits one packing — centers `(xᵢ, yᵢ)` and radii
`rᵢ ≥ 0` — scored by `Σ rᵢ` alone. The radii are **free and unequal**: a good packing mixes a few
large circles with many small gap-fillers, and the optimum is irregular with no symmetry.

## Constraints (feasibility)

- Inside the square: `rᵢ ≤ xᵢ ≤ 1 − rᵢ`, `rᵢ ≤ yᵢ ≤ 1 − rᵢ`.
- Pairwise disjoint: `√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ` for `i ≠ j`.
- `rᵢ ≥ 0`. Accepted when no constraint is violated beyond a small absolute tolerance
  (`atol=1e-6` in the AutoEvolver/OpenEvolve harness; `1e-7` in ShinkaEvolve).

Nonconvex QCQP. Key structural fact: for **fixed centers, the optimal radii are an LP**. The hard,
nonconvex part is *where to put the centers*, in a landscape of many basins of differing quality.

## Prior art and the frontier

Published frontier for `n = 26`: AlphaEvolve `2.63586276` (arXiv:2506.13131, improving Friedman-2012
`~2.634`), ShinkaEvolve `2.635983283` (arXiv:2509.19349), ThetaEvolve `2.63598308`, and the record
AutoEvolver/Claude Code `2.635988438568` (github.com/tengxiaoliu/autoevolver,
tengxiaoliu.github.io/autoevolver) found with ~16.6h autonomous compute. These program-evolution /
agentic systems converged on the same hybrid pipeline: **structured initialization** (golden-angle
spiral + corner/edge seeding), **joint SLSQP refinement** of centers *and* radii together (the
reported key jump came from optimizing centers and radii jointly rather than splitting LP-for-radii
from a separate center optimizer), and **iterated perturbation chains / simulated annealing** that
perturb and re-refine the incumbent to escape local optima.

A grid baseline lands near `2.54`; one SLSQP run near `2.59`; random multi-start saturates near
`2.62`, because uniform scatters rarely seed the rare top-quality basins.

## This method's role

This is the **endpoint**: reproduce the frontier hybrid pipeline within a single bounded
constructor run — structured spiral/corner-seeded restarts, joint SLSQP refinement, then iterated
perturbation chains exploiting the best packing found — pushing into the `2.636` frontier band. The
single editable function is `construct_packing() -> (centers, radii)`.
