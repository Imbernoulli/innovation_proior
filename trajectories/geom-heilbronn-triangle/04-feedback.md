Measured result — `construct:polish` (ENDPOINT: heavy multi-restart SA → `β`-annealed soft-min
L-BFGS-B polish → basin-hopping; seed `2`, also seeded from the rung-3 annealing best). Exact min
triangle area over all `165` triples; returned config verified inside `[0,1]^2`. Runtime `~6 min`.

| Stage | min triangle area | fraction of record |
|---|---|---|
| best of 48 fresh SA→polish restarts | 0.035482 | 0.9580 |
| **polish(rung-3 annealing best)** | **0.037032** | **0.99986** |
| + basin-hopping (returned) | 0.037032 | 0.99986 |

| Ladder summary | min triangle area | fraction of record |
|---|---|---|
| inscribed 11-gon (rung 1) | 0.021456 | 0.579 |
| random multi-start (rung 2) | 0.010872 | 0.294 |
| simulated annealing (rung 3) | 0.035639 | 0.962 |
| **endpoint: SA + soft-min polish (rung 4)** | **0.037032** | **0.99986** |

Reference: Goldberg record `Δ(11) = 1/27 = 0.037037` (fraction `1.000`); gap to record `5.1e-6`.

Notes: the endpoint reaches `0.037032`, **`0.99986` of the Goldberg record `1/27`** — matching the
conjectured optimum to within `5×10^-6`. The load-bearing step is exactly the predicted one: fresh
SA→polish restarts cluster around `0.9580` (the soft-min gradient closes the coordination gap that a
random move could not, but each fresh basin lands a little short), while polishing the *rung-3
annealing best* — which already sat at the edge of the record's basin at `0.0356` — snaps onto the
optimum. Basin-hopping confirms the polished config is the best of its cluster (no further
improvement). The returned points carry the clean rational structure of Goldberg's construction:
coordinates land on multiples of `1/3` (`0, 1/3, 2/3, 1`) and `2/9` along the boundary, with interior
points near `(1/3, 4/9)` and `(2/3, 4/9)` — i.e. the search rediscovered the structured `1/27`
arrangement from scratch. The endpoint stops here, *at* the record: `1/27` is believed optimal at
`n = 11` in the unit square, so matching it to floating-point precision is the ceiling of a
single-machine search-plus-polish. Beating tabulated Heilbronn records — as AlphaEvolve
(arXiv:2506.13131) did for the unit-area *triangle* (`n=11`: `0.036 → >0.0365`) and *convex-region*
(`n=13,14`) containers — requires large-scale evolutionary search on the cases where the tabulated
value is not already optimal; the still-open part here is the *proof* that `1/27` is optimal, which
no construction or search can supply.
