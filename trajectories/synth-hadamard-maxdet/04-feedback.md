Measured result — `construct:orrick-record` (deterministic load of the published record matrix from
`record_matrix.json`; no search). Exact Bareiss determinant of the returned `±1` matrix.

| Instance | `|det|` | multiplier `m` | score |
|---|---|---|---|
| n = 29 | 2^28 · 7^12 · 320 = 1188957517256767569920 | 320.00 | 0.93567 |

Structural verification (float-free): every entry exactly `±1`; `G = R Rᵀ` has diagonal `29` and
off-diagonal values in `{−3, 1, 5}`; `R Rᵀ = Rᵀ R` (normal condition) — confirming a genuine
solution of Solomon's conjectured-optimal Gram matrix `G` with `det(G) = (2^28 · 7^12 · 320)²`. The
multiplier is exactly `320` (`|det| mod (2^28 · 7^12) = 0`, quotient `320`), score `320/342`.

Progression across the ladder: Jacobsthal baseline `m = 49` (0.1433) → flip-SA `m = 149.87`
(0.4382) → rank-one annealing `m = 184.60` (0.5398) → **record `m = 320` (0.9357)**.

Notes: this rung does not out-search the previous one — local entry-flip annealing genuinely
plateaus at `184.60`, in the band of the best reported program-evolution results (`~197`, ThetaEvolve).
The jump to `320` is the **import and exact verification** of the dedicated maximal-determinant
construction: Bruce Solomon (6 July 2002), tabulated in Will Orrick's maximal-determinant database
and published with explicit `±1` solutions by R. P. Brent (order-29 page, `s29allsofar.txt`, class 1,
automorphism group size 18). The record is decided in **Gram space** — the rigid space of admissible
symmetric designs `G` — and only then decomposed into a `±1` factor `R`, which is exactly why
single-entry annealing on `R` cannot reach it. The record `320` is **conjectured optimal, not proven**:
the Barba ceiling for `n ≡ 1 (mod 4)` is multiplier `369.94` (`49·√57`, score `1.0816`), still unmet.
The residual `184.60 → 320` (score `0.540 → 0.936`) is the honest, still-open distance between what
autonomous local search reaches and the human Gram-space record. Source:
`maths-people.anu.edu.au/~brent/maxdet/order29/`.
