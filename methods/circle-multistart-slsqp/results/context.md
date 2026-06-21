# Context: sum-of-radii circle packing in the unit square (n = 26)

## Research question

Place `26` circles inside the unit square `[0,1]²`, pairwise non-overlapping, and maximize the
**sum of their radii** `Σ rᵢ`. The constructor emits one packing — centers `(xᵢ, yᵢ)` and radii
`rᵢ ≥ 0` — scored by `Σ rᵢ` alone. The radii are **free and unequal**, so a packing can mix
large circles with small gap-fillers. The broad question is how to choose the `26` center positions
that yield the largest achievable `Σ rᵢ`.

## Constraints (feasibility)

- Inside the square: `rᵢ ≤ xᵢ ≤ 1 − rᵢ`, `rᵢ ≤ yᵢ ≤ 1 − rᵢ`.
- Pairwise disjoint: `√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ` for `i ≠ j`.
- `rᵢ ≥ 0`. Accepted when no constraint is violated beyond a small absolute tolerance.

This is a nonconvex QCQP. Structural fact: for **fixed centers, the optimal radii are an LP** —
maximize `Σ rᵢ` s.t. `rᵢ + rⱼ ≤ dᵢⱼ`, `rᵢ ≤ wallᵢ`. The nonconvex part is *where to put the
centers*, and the landscape has many local basins of differing quality.

## Where this task sits

Published frontier for `n = 26`: AlphaEvolve `2.63586276` (arXiv:2506.13131), ShinkaEvolve
`2.635983283` (arXiv:2509.19349), ThetaEvolve `2.63598308`, record AutoEvolver/Claude Code
`2.635988438568` (github.com/tengxiaoliu/autoevolver).

## Baselines

- **Grid baseline.** Place centers on a regular grid and tighten radii; lands near `2.54`.
- **Single SLSQP.** From one random center scatter, jointly refine centers and radii with SLSQP
  (sequential least-squares quadratic programming), enforcing the pairwise and wall inequality
  constraints, then re-tighten the radii to their LP optimum. A single run lands near `2.59`,
  settling into one local basin of the nonconvex landscape.

The single editable function is `construct_packing() -> (centers, radii)`.
