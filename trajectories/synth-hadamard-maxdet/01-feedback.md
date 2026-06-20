Measured result — `construct:jacobsthal` (`R = Q + I`), exact integer determinant via Bareiss
elimination. Deterministic (no seed).

| Instance | `|det|` | multiplier `m` | score |
|---|---|---|---|
| n = 29 | 2^28 · 7^12 · 49 | 49.00 | 0.14330 |

Reference points: classical record `m = 320` (score 0.9357), Barba ceiling `m = 369.94` (score
1.0816), best reported LLM-evolution `m ≈ 197` (score ≈ 0.576).

Notes: the multiplier is exactly `49 = 7^2`, so `|det| = 2^28 · 7^14`, the clean closed-form value
of the symmetric Jacobsthal design — and it coincides exactly with the published task baseline
(score `0.1433`), confirming the scaffold default is this construction. `Q + I` and `Q − I` give
identical `|det|` as predicted. The structure is rigid: every single-entry change *lowers* `|det|`,
so this point is a strict local maximum under entry flips. The whole gap to the record (`49 → 320`)
must be bought by search that leaves the symmetry — which is the next rung.
