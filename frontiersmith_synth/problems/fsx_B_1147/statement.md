# Water Rations Set Before the Weather Book Opens

## Problem

A farm runs `F` fields over a `T`-week season with one shared irrigation
budget. You must commit to a single weekly irrigation schedule for every
field **before** the weather is known. Afterwards, a "weather book" of `K`
equally-plausible rainfall scenarios is opened, and your ONE fixed schedule
is graded against ALL `K` of them.

Each field `f` has a soil-moisture recurrence. Starting from moisture
`m0[f]`, each week `t` (0-indexed) it updates as
```
m <- clip(m + X[f][t] + rain[k][t] - cons[f], 0, cap[f])
```
where `X[f][t]` is the water you irrigate field `f` with in week `t`,
`rain[k][t]` is that week's rainfall under scenario `k`, `cons[f]` is the
crop's weekly consumption, and `clip` clamps to `[0, cap[f]]` (moisture
cannot go negative, and excess above capacity runs off and is wasted).

Growth accrues weekly. In a week where `m >= thresh[f]` (the crop's stress
threshold) the field gains
```
rate[f]*thresh[f] + (rate[f] * min(m - thresh[f], thresh[f]//2)) // 4
```
that week: a **flat** reward for simply clearing the threshold (the
dominant term -- staying unstressed is what matters), plus a small, capped
**bonus** for extra moisture above threshold (integer division; a modest,
secondary reward for spending any surplus budget well). In a week where
`m < thresh[f]` the crop is stressed and gains **0** yield that week (it
can still recover and resume growing later if moisture climbs back above
threshold).

For scenario `k`, `scenario_yield(k)` is the sum of every field's
season-total yield under that scenario's rainfall (using the *same* `X` you
committed to). The final objective is the **product** over all `K`
scenarios:
```
OBJECTIVE = scenario_yield(1) * scenario_yield(2) * ... * scenario_yield(K)
```
Because this is a product, a single scenario in which several fields spend
many weeks stressed contributes a small factor that drags the *entire*
objective down, no matter how well the other scenarios did.

## Input (stdin)
```
T F K
TotalBudget WeeklyCap
m0_1 cap_1 cons_1 thresh_1 rate_1
...
m0_F cap_F cons_F thresh_F rate_F
rain_1[0] rain_1[1] ... rain_1[T-1]
...
rain_K[0] rain_K[1] ... rain_K[T-1]
```
All values are non-negative integers.

## Output (stdout)
```
F T
X_1[0] X_1[1] ... X_1[T-1]
...
X_F[0] X_F[1] ... X_F[T-1]
```
Print `F` and `T` first (must match the input), then one line per field
with `T` non-negative integers: the irrigation to apply to that field in
each week. This single schedule is what gets simulated against every one
of the `K` scenarios.

## Feasibility
- The header `F T` must exactly match the input dimensions.
- Exactly `F*T` further integer tokens must follow (no more, no fewer).
- Every irrigation value must be a finite non-negative integer.
- For every week `t`, the cross-field sum `sum_f X[f][t]` must not exceed
  `WeeklyCap` (a shared delivery-pipe limit -- the season budget must be
  split across time, you cannot deliver it all in one week).
- The grand total `sum_{f,t} X[f][t]` must not exceed `TotalBudget`.

Any violation scores `Ratio: 0.0`.

## Scoring
The checker builds its own feasible reference schedule (split `TotalBudget`
evenly across every field-week slot, respecting `WeeklyCap`), computes the
same product-of-scenario-yields objective `B` on it, and scores
```
Ratio = min(1.0, OBJECTIVE / (10 * B))
```
i.e. `Ratio = min(1.0, 0.1 * OBJECTIVE / B)`, printed as `Ratio: <value>`.
A schedule that keeps every field above its stress threshold in every
scenario typically multiplies far more scenario factors together at full
strength than the uniform baseline (which usually leaves some scenario
badly stressed), so the two are not close.

## Constraints
`3 <= F <= 8`, `8 <= T <= 20`, `3 <= K <= 8`. Time limit 5s, memory 512MB.

## Example (worked score, small illustrative instance)
`T=2, F=1, K=2`, field: `m0=10 cap=20 cons=4 thresh=8 rate=1`
(`half_thresh=4`), `TotalBudget=6, WeeklyCap=6`, `rain_1=[0,0]`,
`rain_2=[4,4]`.
Schedule `X=[4,0]`: scenario 1, week0 `m=10+4+0-4=10>=8` -> margin
`min(2,4)=2`, yield `1*8+(1*2)//4=8`; week1 `m=10+0+0-4=6<8` ->
stressed, yield 0; `scenario_yield(1)=8`.
Scenario 2, week0 `m=10+4+4-4=14` -> margin `min(6,4)=4`, yield
`1*8+(1*4)//4=9`; week1 `m=14+0+4-4=14` -> yield 9 again;
`scenario_yield(2)=18`.
`OBJECTIVE=8*18=144`. This illustrates the mechanics only; real
instances use larger `F, T, K` with several weather scenarios per test.
