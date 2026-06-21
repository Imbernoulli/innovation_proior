# Context: sum-of-radii circle packing in the unit square (n = 26)

## Research question

Place `26` circles inside the unit square `[0,1]²`, pairwise non-overlapping, and maximize the
**sum of their radii** `Σ rᵢ`. The constructor emits one packing — centers `(xᵢ, yᵢ)` and radii
`rᵢ ≥ 0` — scored by `Σ rᵢ` alone. The radii are **free and unequal**: each `rᵢ` is an
independent decision variable, not fixed in advance.

## Constraints (feasibility)

- Inside the square: `rᵢ ≤ xᵢ ≤ 1 − rᵢ`, `rᵢ ≤ yᵢ ≤ 1 − rᵢ`.
- Pairwise disjoint: `√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ` for `i ≠ j`.
- `rᵢ ≥ 0`. A packing is accepted when no constraint is violated by more than a small absolute
  tolerance.

This is a nonconvex QCQP (the pairwise-distance constraints are nonconvex). A structural fact:
for **fixed centers, the optimal radii are an LP** — maximize `Σ rᵢ` s.t. `rᵢ + rⱼ ≤ dᵢⱼ`,
`rᵢ ≤ wallᵢ`, where `dᵢⱼ` is the center-to-center distance and `wallᵢ` is the distance to the
nearest wall.

## Where this task sits

The published frontier for `n = 26`: AlphaEvolve `2.63586276` (arXiv:2506.13131, improving
Friedman-2012 `~2.634`), ShinkaEvolve `2.635983283` (arXiv:2509.19349), ThetaEvolve `2.63598308`,
and the record AutoEvolver/Claude Code `2.635988438568` (github.com/tengxiaoliu/autoevolver). A
trivial structured baseline lands near `2.54`; the frontier band is a few parts in the sixth
decimal.

## This method's role

This is the **structured baseline**: a parameter-free, closed-form, guaranteed-feasible layout
that puts a concrete floor on the board. The single editable function is
`construct_packing() -> (centers, radii)`.
