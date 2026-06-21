# Context: sum-of-radii circle packing in the unit square (n = 26)

## Research question

Place `26` circles inside the unit square `[0,1]²`, pairwise non-overlapping, and maximize the
**sum of their radii** `Σ rᵢ`. The constructor emits one packing — centers `(xᵢ, yᵢ)` and radii
`rᵢ ≥ 0` — scored by `Σ rᵢ` alone. The radii are **free and unequal**: a packing may mix a few
large circles with many small gap-fillers, and the optimum carries no imposed symmetry.

## Constraints (feasibility)

- Inside the square: `rᵢ ≤ xᵢ ≤ 1 − rᵢ`, `rᵢ ≤ yᵢ ≤ 1 − rᵢ`.
- Pairwise disjoint: `√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ` for `i ≠ j`.
- `rᵢ ≥ 0`. Accepted when no constraint is violated beyond a small absolute tolerance.

This is a nonconvex QCQP (the pairwise-distance constraints are nonconvex). Structural fact:
for **fixed centers, the optimal radii are an LP** — maximize `Σ rᵢ` s.t. `rᵢ + rⱼ ≤ dᵢⱼ`,
`rᵢ ≤ wallᵢ`.

## Where this task sits

Published frontier for `n = 26`: AlphaEvolve `2.63586276` (arXiv:2506.13131), ShinkaEvolve
`2.635983283` (arXiv:2509.19349), ThetaEvolve `2.63598308`, record AutoEvolver/Claude Code
`2.635988438568` (github.com/tengxiaoliu/autoevolver). A structured grid baseline lands near
`2.54`.

## This method's role

The task is to produce a feasible packing for `n = 26` that improves on the rigid grid baseline by
treating placement as a continuous optimization over the `78` decision variables (`26` centers plus
`26` radii) subject to the wall and pairwise-distance constraints above. The single editable
function is `construct_packing() -> (centers, radii)`, returning the `26` centers and radii of one
packing.
