Measured result — `construct:goldberg-optimal` (ENDPOINT: Goldberg's exact rational `n=11`
configuration, no search). Minimum triangle area verified in **exact rational arithmetic** over all
`165` triples, then re-confirmed in double precision against the harness evaluator. Runtime `<0.1 s`.

| Stage | min triangle area | fraction of record |
|---|---|---|
| previous endpoint (SA + soft-min polish, rung 4) | 0.037032 | 0.99986 |
| **Goldberg exact configuration (rung 5)** | **1/27 = 0.0370370370** | **1.000000** |

| Ladder summary | min triangle area | fraction of record |
|---|---|---|
| inscribed 11-gon (rung 1) | 0.021456 | 0.579 |
| random multi-start (rung 2) | 0.010872 | 0.294 |
| simulated annealing (rung 3) | 0.035639 | 0.962 |
| SA + soft-min polish (rung 4) | 0.037032 | 0.99986 |
| **endpoint: Goldberg exact configuration (rung 5)** | **1/27 = 0.0370370370** | **1.000000** |

Reference: Goldberg record `Δ(11) = 1/27 = 0.0370370370` (Goldberg 1972; Erich Friedman's
Heilbronn-for-squares page lists `1/27`, "horizontally symmetric"; exact rational coordinates tabulated
in arXiv:2603.11107 Table 11). `1/27` is the **conjectured optimal** value at `n = 11` in the unit
square — believed but not proven. Gap to record: **exactly `0`** (the rational min equals `1/27`; the
double-precision min agrees to `1.4×10^-17`, machine epsilon).

Notes: the rung closes the previous `5×10^-6` residual — which was pure optimizer tolerance, not a
geometric gap — by reproducing Goldberg's exact configuration rather than approximating it. The exact
evaluator (each coordinate a `Fraction`, each area half an integer-denominator cross product, no
rounding) returns the literal `1/27`, with **`28` of the `165` triangles tied at exactly the minimum** —
the rigid, over-determined binding web that pins the layout as a genuine max-min optimum. The
configuration is mirror-symmetric about `x = 1/2`, with eight points on the boundary and three interior,
all on simple fractions (denominators `3, 9, 6, 2`); the rung-4 search was groping toward exactly this
arrangement (it had already found points on multiples of `1/3`, `2/9`, `4/9`). The ladder stops here,
*at* the record: matching `1/27` is the ceiling, because `1/27` is believed optimal at `n = 11`, so the
still-open part is the *proof* of optimality, which no construction or search can supply. Neighboring
records for orientation: `Δ(10) ≈ 0.0465` (Comellas–Yebra), `Δ(12) ≈ 0.0326` (Comellas–Yebra 2001),
`Δ(13) ≈ 0.0270` (Karpov 2011) — `n = 11`'s clean `1/27` sits between the messier `n=12` and `n=10`
records.
