# Context: sum-of-radii circle packing in the unit square (n = 26)

## Research question

Place `26` circles inside the unit square `[0,1]²`, pairwise non-overlapping, and maximize the
**sum of their radii** `Σ rᵢ`. The constructor emits one packing — centers `(xᵢ, yᵢ)` and radii
`rᵢ ≥ 0` — scored by `Σ rᵢ` alone. The radii are **free and unequal**: a packing may mix a few
large circles with many small gap-fillers, and the optimum is irregular with no imposed symmetry.

## Constraints (feasibility)

- Inside the square: `rᵢ ≤ xᵢ ≤ 1 − rᵢ`, `rᵢ ≤ yᵢ ≤ 1 − rᵢ`.
- Pairwise disjoint: `√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ` for `i ≠ j`.
- `rᵢ ≥ 0`. Accepted when no constraint is violated beyond a small absolute tolerance
  (`atol=1e-6` in the AutoEvolver/OpenEvolve harness; `1e-7` in ShinkaEvolve).

This is a nonconvex QCQP. Structural fact: for **fixed centers, the optimal radii are an LP** — the
remaining degrees of freedom are *where to put the centers*, in a landscape of many basins of
differing quality.

## Prior art and the frontier

Published values for `n = 26`: Friedman-2012 `~2.634`, AlphaEvolve `2.63586276` (arXiv:2506.13131),
ShinkaEvolve `2.635983283` (arXiv:2509.19349), ThetaEvolve `2.63598308`, and the record
AutoEvolver/Claude Code `2.635988438568` (github.com/tengxiaoliu/autoevolver,
tengxiaoliu.github.io/autoevolver) found with ~16.6h autonomous compute.

A grid baseline lands near `2.54`; one SLSQP run lands near `2.59`; random multi-start saturates
near `2.62`.

## This method's role

This is the **endpoint**: a single bounded constructor run that targets the `2.636` frontier band.
The single editable function is `construct_packing() -> (centers, radii)`.
