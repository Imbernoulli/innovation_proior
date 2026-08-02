# Credit Limit Assignment Under a Portfolio Loss Cap

## Problem
A card issuer has `N` applicants. For applicant `i` you must choose ONE credit-limit tier
from a fixed increasing list `L[0]=0 < L[1] < ... < L[M]` (dollars); tier `0` means "decline".
For every applicant and every nonzero tier, historical data (from past cohorts who were actually
given that tier) supplies two numbers that would result if you assigned that tier:
`default_bps[i][m]` (probability of default, basis points, 0-10000) and `util_bps[i][m]`
(the fraction of the limit, basis points, actually carried as an average outstanding balance).
Both depend on the **tier itself**, not just the applicant's baseline credit score: handing out a
bigger limit can raise default probability (over-extension), and it can change how much of the
limit gets used -- some applicants plateau or pull back once given more room than they need,
others keep spending more the more they are given.

For applicant `i` assigned tier `m >= 1`, using the instance's APR (bps) and loss severity `SEV`
(bps, the fraction of a defaulted balance that is lost):
```
balance  = L[m] * util_bps[i][m] // 10000
revenue  = balance * APR // 10000 * (10000 - default_bps[i][m]) // 10000
loss     = balance * default_bps[i][m] // 10000 * SEV // 10000
profit   = revenue - loss
```
(`//` is integer floor division; tier `0` gives balance = revenue = loss = profit = 0.)

Choose exactly one tier per applicant so the portfolio's total expected loss does not exceed a
hard cap `CAP`. Maximize total profit.

## Input (stdin)
```
N M
L[1] L[2] ... L[M]
APR SEV CAP
score_1 d_1_1 u_1_1 d_1_2 u_1_2 ... d_1_M u_1_M
...
score_N d_N_1 u_N_1 ... d_N_M u_N_M
```
`score_i` is applicant `i`'s baseline credit score -- informational only, it is NOT part of the
profit formula. `d_i_m`/`u_i_m` are `default_bps[i][m]`/`util_bps[i][m]`.

## Output (stdout)
`N` integers `m_1 ... m_N` (whitespace/newline separated), each in `[0, M]`: the tier assigned to
applicant `i`, in input order.

## Feasibility
- Exactly `N` tokens, each parses as an integer in `[0, M]`.
- `sum_i loss(i, m_i) <= CAP`.
Any violation (wrong token count, non-integer/out-of-range tier, or a broken cap) scores
`Ratio: 0.0`.

## Objective (maximize)
`F = sum_i profit(i, m_i)`.

## Scoring
The checker's own internal baseline `B` assigns every applicant a fixed tier `2`, processed in
input order, skipping an applicant once doing so would break `CAP`, and sums `profit`. With your
feasible `F`:
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so matching the baseline scores about `0.1`, and a portfolio ten times more profitable caps at
`1.0`. A non-positive `F` (or degenerate `B`) scores `0.0`.

## Constraints
`10 <= N <= 60`, `M = 6`, `1 <= L[1] < ... < L[6] <= 20000`, all `bps` values in `[0,10000]`,
`CAP >= 0`. Runs in well under the time limit for these sizes.

## Example
`N=2, M=6`, tiers `800 1600 3200 6000 11000 18000`, `APR=1500 SEV=5000 CAP=100`.
Applicant 1 (`score=700`): `d=(200,260,340,460,620,820) u=(6000,6300,6500,6600,6650,6680)` for
tiers `1..6`, giving `(profit,loss)` = `(66,4) (134,13) (266,35) (475,91) (802,226) (1163,492)`.
Applicant 2 (`score=500`): `d=(300,410,650,1200,2400,4400) u=(5500,6000,6600,7200,8000,8600)`,
giving `(58,6) (119,19) (227,68) (311,259) (-53,1056) (-2105,3405)` -- its own profit-maximizing
tier is `4`, not the top tier, because default risk accelerates faster than revenue past it.
Under `CAP=100`, assigning applicant 1 tier `4` (loss `91`) and applicant 2 tier `1` (loss `6`)
gives total loss `97 <= 100` and `F = 475 + 58 = 533`. The baseline `B` (both at tier `2`, total
loss `13+19=32 <= 100`) gives `B = 134 + 119 = 253`. `Ratio = min(1000, 100*533/253)/1000 =
0.2107`.
