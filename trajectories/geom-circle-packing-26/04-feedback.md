Measured result — `construct_packing:structured-perturb-slsqp` (golden-angle-spiral + corner-seeded
structured restarts → joint SLSQP → iterated perturbation chains; seed `7`, budget `520 s`).
Feasibility verified at `atol=1e-7`.

| Instance | seed | budget | Σ rᵢ | feasible | max constraint violation |
|---|---|---|---|---|---|
| n = 26 | 7 | 520 s | 2.6274899713 | yes | 6.06e-12 |

Reference points: grid `2.5414`, single-SLSQP `2.5949`, multistart-SLSQP `2.6221`; AlphaEvolve
`2.63586276`, ShinkaEvolve `2.635983283`, ThetaEvolve `2.63598308`, AutoEvolver record
`2.635988438568`.

Notes: the hybrid pipeline lifts the sum `+0.0054` over random multi-start (`2.6221 → 2.6275`). The
phase trace shows where the gains come from — structured spiral/corner restarts reach `~2.6248`
quickly, then iterated perturbation chains mine the incumbent up to `2.62749`, exactly the
structure-then-memory behavior the method was built for. The returned packing is genuinely feasible
(`maxviol ≈ 6e-12`, far inside both the `1e-7` and `1e-6` harness tolerances). This lands in the
frontier *neighborhood* but `~0.0085` below the AutoEvolver record `2.635988438568`: the published
frontier values are separated by only parts in the sixth decimal and were found with orders of
magnitude more search (the record used ~16.6 h of autonomous compute) — this is the right pipeline
run under a bounded `~9 min` budget, so the residual gap is bought with search budget, not a better
algorithm. The endpoint of the ladder is this frontier-neighborhood value; the record stands above
as the part of the problem that sustained search, not a new construction, would close. Endpoint
informed by AutoEvolver (github.com/tengxiaoliu/autoevolver) and ShinkaEvolve (arXiv:2509.19349):
joint centers+radii SLSQP + structured init + iterated perturbation chains.
