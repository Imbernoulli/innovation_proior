Measured result — `construct_packing:single-slsqp` (one SLSQP run on the full `78`-variable
problem from one random start, seed `0`, radii LP-re-tightened). Feasibility verified at
`atol=1e-7`.

| Instance | seed | Σ rᵢ | feasible | max constraint violation |
|---|---|---|---|---|
| n = 26 | 0 | 2.5949163422 | yes | 1.22e-14 |

Reference points: grid baseline `2.5414213562`; AlphaEvolve `2.63586276`, ShinkaEvolve
`2.635983283`, AutoEvolver record `2.635988438568`.

Notes: a single SLSQP run lifts the sum `+0.0535` over the grid (`2.5414 → 2.5949`) by making the
radii unequal and sliding circles into the corners and edges the grid wasted — the optimizer is
doing the right thing. But it is still `~0.041` below the frontier, because one random
initialization lands in one local basin of a nonconvex landscape with many basins of differing
quality. The SLSQP engine is correct; the bottleneck is the single start. The next rung wraps the
same solver in many random restarts and keeps the best feasible packing.
