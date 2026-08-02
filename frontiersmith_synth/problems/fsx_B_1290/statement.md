# Payment Rail Cascades: Cost Per Successful Charge

A merchant can route a card charge over several payment **rails** (processors).
Each rail charges an **interchange fee** that depends on the ticket size, and
each rail **authorizes** (succeeds) with a probability that depends on the
**issuer segment** of the customer's card -- some rails work great for one
segment and poorly for another. If a rail **declines** the charge, you may
retry on a *different* rail; every retry attempt pays its own interchange fee
plus a rail-specific **retry surcharge** (re-authorization overhead). After a
fixed cap of attempts, an unrecovered decline is a **lost payment**, costing a
fixed penalty. You must decide, for every (issuer segment, ticket-size bucket)
pair, an ordered **cascade** of distinct rails to try.

The illustrative example below uses toy numbers -- not the real scoring
constants, which live entirely in the input.

## Input (stdin)
```
R S B K FAILPEN_BPS
amt_0 amt_1 ... amt_{B-1}
fixedFee_0_0 pctBps_0_0 ... fixedFee_0_{B-1} pctBps_0_{B-1} retrySurcharge_0
...                                                              (R lines, one per rail)
auth_0_0 auth_0_1 ... auth_0_{R-1}
...                                                              (S lines, one per issuer segment)
vol_0_0 vol_0_1 ... vol_0_{B-1}
...                                                              (S lines, one per issuer segment)
```
`R` rails (0-indexed), `S` issuer segments, `B` ticket-size buckets, `K` = max
cascade length (attempts). `amt[b]` is the representative ticket size (cents)
for bucket `b`; a lost payment on bucket `b` costs `failpen(b) = FAILPEN_BPS *
amt[b] / 10000` cents (a fixed FRACTION of the ticket value, so the penalty
scales sanely across ticket sizes instead of one flat constant). For rail
`r`, bucket `b`: `fee(r,b) = fixedFee[r][b] + pctBps[r][b] * amt[b] / 10000`
(cents) -- the interchange-fee tier. `retrySurcharge[r]` is the extra cost
charged on rail `r` whenever it is used as a **non-first** attempt in a
cascade. `auth[s][r]` in `[0,1]` is the probability rail `r` authorizes a
charge from issuer segment `s` (independent across attempts). `vol[s][b]` is
the transaction volume for segment `s`, bucket `b`.

## Output (stdout)
Exactly `S*B` lines, one per `(segment, bucket)` pair in row-major order
(`s=0..S-1` outer, `b=0..B-1` inner): `L r_1 r_2 ... r_L` -- a cascade of `L`
**distinct** rail ids (`1 <= L <= K`), tried in order until one authorizes or
the cascade is exhausted.

## Feasibility
For every cell: `1 <= L <= K`, every `r_i` in `[0,R)`, no repeated rail
within a cell. Any violation, truncation, or malformed token scores `0`.

## Objective & Scoring
For cascade `(r_1,...,r_L)` on cell `(s,b)`, let `reach_i` = probability all
of `r_1..r_{i-1}` declined (`reach_1=1`). Expected cost:
`cost = sum_i reach_i * (fee(r_i,b) + retrySurcharge[r_i] if i>1 else fee(r_i,b)) + reach_{L+1} * failpen(b)`,
where `reach_{L+1}` is the probability every attempt in the cascade declined.
Success probability is `succ = 1 - reach_{L+1}`. The cell's **cost per
successful payment** is `cost / succ`. Your total score is the
`vol`-weighted mean of this ratio over all `S*B` cells: `F`.

The checker also builds a baseline `B` = the cost-per-success of routing
*every* cell to a single fixed attempt on whichever rail has the lowest
average fee (no cascade at all). Since this is a MINIMIZE objective:
```
Ratio = min(1.0, 0.1 * B / F)
```
so the naive single-cheap-rail baseline scores `~0.1`; a routing policy with
a genuinely lower cost-per-success climbs toward `1.0`.

## What makes it hard
Routing to the cheapest fee lowers cost *per attempt*, but if that rail
authorizes poorly for a given issuer segment, most of its attempts are
wasted -- and the eventual retry (on a pricier rail) still has to pay its own
fee plus a retry surcharge, or the payment is lost outright. The right
cascade -- which rails to use, in what order, and how many -- can differ
sharply *per issuer segment*, since auth rates are segment-specific while
fees are not. Minimizing cost per **attempt** and minimizing cost per
**successful payment** are different objectives, and only the second is
scored.

## Constraints
`3 <= R <= 6`, `2 <= S <= 6`, `B = 3`, `2 <= K <= 4`. Time limit 5s, memory
512MB. Scoring is deterministic.
