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

This is a nonconvex QCQP. Key structural fact: for **fixed centers, the optimal radii are an LP** —
maximize `Σ rᵢ` s.t. `rᵢ + rⱼ ≤ dᵢⱼ`, `rᵢ ≤ wallᵢ`. The hard, nonconvex part is *where to put the
centers*, and the landscape has many local basins of differing quality.

## Where this task sits

Published frontier for `n = 26`: AlphaEvolve `2.63586276` (arXiv:2506.13131), ShinkaEvolve
`2.635983283` (arXiv:2509.19349), ThetaEvolve `2.63598308`, record AutoEvolver/Claude Code
`2.635988438568` (github.com/tengxiaoliu/autoevolver). A grid baseline lands near `2.54`; a single
SLSQP run near `2.59`.

## This method's role

This is the **multi-start** method: wrap SLSQP in many random restarts and keep the best feasible
packing. Each refined start is a distinct local optimum, and the best-of-many is an order statistic
that climbs with the number of starts — converting compute into quality. It beats single-start but
plateaus below the frontier, because uniform random scatters rarely seed the rare top-quality
basins. The single editable function is `construct_packing() -> (centers, radii)`.
