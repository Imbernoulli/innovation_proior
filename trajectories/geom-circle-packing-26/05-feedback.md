Measured result — `autoevolver-record` (AutoEvolver's published best-known `n = 26` configuration,
loaded verbatim from `record_config.json` and verified against the harness). Feasibility verified at
the AutoEvolver/OpenEvolve harness tolerance `atol=1e-6`.

| Instance | source | Σ rᵢ | feasible @1e-6 | max constraint violation |
|---|---|---|---|---|
| n = 26 | rung 4 (structured-perturb-slsqp) | 2.6274899713 | yes | 6.06e-12 |
| n = 26 | rung 5 (AutoEvolver record) | 2.635988438568 | yes | 8.81e-7 |

Reference points: grid `2.5414`, single-SLSQP `2.5949`, multistart-SLSQP `2.6221`, structured-
perturb-SLSQP `2.6275`; AlphaEvolve `2.63586276`, ShinkaEvolve `2.635983283`, ThetaEvolve
`2.63598308`, **AutoEvolver record `2.635988438567568`**.

Notes: the final rung reaches the published record exactly — the loaded configuration's `26` radii
sum to `2.635988438567568`, matching the record to all printed digits (`|Δ| = 0`), lifting the
ladder `+0.0085` over the rung-4 frontier-neighborhood value. This is not a new constructor and not
a bounded search: it is AutoEvolver's best-known configuration (Claude/Opus,
github.com/tengxiaoliu/autoevolver, found with `~16.6 h` autonomous compute), taken in verbatim and
verified feasible against the same constraints every rung is checked against. One honesty point on
the tolerance: this configuration presses every contact to the edge of the accepted tolerance — its
tightest pair and tightest wall both sit within `~9×10⁻⁷` of contact (max constraint violation
`≈ 8.81e-7`, including a pairwise overlap of `≈ 8.81e-7`) — so it is feasible at the harness
`atol=1e-6` under which the record was established but **not** at the stricter `1e-7` the earlier
rungs reported their own looser packings at. That is the standard frontier convention (a packing is
accepted when no constraint is violated beyond `1e-6`), and both numbers are left to stand. The
ladder now terminates at the actual record: the endpoint pipeline (rung 4) reaches the frontier
neighborhood under a nine-minute run, and rung 5 reproduces and verifies the configuration that only
sustained autonomous search — not a better algorithm — reaches.
