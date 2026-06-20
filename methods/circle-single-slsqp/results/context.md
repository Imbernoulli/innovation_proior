# Context: sum-of-radii circle packing in the unit square (n = 26)

## Research question

Place `26` circles inside the unit square `[0,1]²`, pairwise non-overlapping, and maximize the
**sum of their radii** `Σ rᵢ`. The constructor emits one packing — centers `(xᵢ, yᵢ)` and radii
`rᵢ ≥ 0` — scored by `Σ rᵢ` alone. The radii are **free and unequal**: a good packing mixes a few
large circles with many small gap-fillers, and the optimum is irregular with no symmetry.

## Constraints (feasibility)

- Inside the square: `rᵢ ≤ xᵢ ≤ 1 − rᵢ`, `rᵢ ≤ yᵢ ≤ 1 − rᵢ`.
- Pairwise disjoint: `√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ` for `i ≠ j`.
- `rᵢ ≥ 0`. Accepted when no constraint is violated beyond a small absolute tolerance.

This is a nonconvex QCQP (the pairwise-distance constraints are nonconvex). Key structural fact:
for **fixed centers, the optimal radii are an LP** — maximize `Σ rᵢ` s.t. `rᵢ + rⱼ ≤ dᵢⱼ`,
`rᵢ ≤ wallᵢ`. The hard, nonconvex part is *where to put the centers*.

## Where this task sits

Published frontier for `n = 26`: AlphaEvolve `2.63586276` (arXiv:2506.13131), ShinkaEvolve
`2.635983283` (arXiv:2509.19349), ThetaEvolve `2.63598308`, record AutoEvolver/Claude Code
`2.635988438568` (github.com/tengxiaoliu/autoevolver). A structured grid baseline lands near
`2.54`.

## This method's role

This is the **single-start nonlinear-optimization** method: formulate the full `78`-variable
problem (centers + radii) and run one SLSQP refinement from a single random initialization, with
the radii re-tightened to their LP optimum. It establishes what one local descent on the true
constrained problem reaches — beating the rigid grid by making radii unequal and using the
corners/edges — and exposes that the bottleneck is the lone initialization, not the solver. The
single editable function is `construct_packing() -> (centers, radii)`.
