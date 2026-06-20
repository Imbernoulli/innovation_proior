Measured result — `construct_packing:grid-baseline` (`5×5` equal-circle grid + one interstitial
filler). Deterministic (no seed). Feasibility verified against the fixed checker.

| Instance | Σ rᵢ | feasible | max constraint violation |
|---|---|---|---|
| n = 26 | 2.5414213562 | yes | 5.55e-17 |

Reference points: AlphaEvolve `2.63586276`, ShinkaEvolve `2.635983283`, AutoEvolver record
`2.635988438568`. Prior human best (Friedman 2012) `≈ 2.634`.

Notes: the sum is exactly `2.5 + (√2 − 1)·0.1 = 2.54142135623…`, the closed-form value of the
`5×5` grid (`25 × 0.1 = 2.5`) plus the inscribed interstitial circle `(√2 − 1)·0.1`. Feasibility
is at the floating-point floor — the `25` grid circles are tangent to neighbours and walls, the
filler tangent to its four neighbours, no overlap beyond `~6×10⁻¹⁷`. This is `0.094` below the
frontier band. The layout is rigid: every grid circle is locked to the same radius `0.1` and the
same lattice, so the corners/edges are underused and no circle can grow. The whole gap to the
frontier must be bought by search that makes the radii unequal and moves the centers — the next
rung hands the problem to constrained nonlinear optimization (SLSQP).
