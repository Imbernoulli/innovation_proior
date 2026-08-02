# Storm-Correlated Underwriting

## Problem
You are steering a hurricane insurance book. You are given `N` candidate policies and
`K` storm scenarios. Each candidate has a location, an exposure (insured amount), a
premium you would charge, and a technical price (the modelled expected loss from
*ordinary*, uncorrelated perils — fire, theft, and so on — priced in isolation). A
policy is inside a storm's **footprint** if it lies within that storm's radius of its
center. Writing a policy earns its premium and costs its technical price regardless of
storms; on top of that, every storm that hits imposes a **catastrophe loss** on the
total exposure your book has accumulated inside that storm's footprint.

You may write at most `C` policies (your underwriting capacity for the period — you
cannot bind all adequately-priced business that walks in the door). Output the set you
write. A larger, more diversified book earns more premium, but exposure that piles up
inside one storm's footprint gets expensive fast once it passes that storm's
**accumulation limit**: below the limit, loss accrues at the storm's normal rate; past
the limit, the *same* extra unit of exposure is charged `OVER_MULT` times that rate
(representing the cost of capital / reinsurance you must buy once your modelled
capacity is exceeded). Because that penalty only switches on once the footprint is
already full, an identical candidate policy can be worth a lot early and worth little
— or negative — once the storms it touches are already crowded by policies you wrote
earlier in the same book.

## Input (stdin)
```
N C K OVER_MULT
K lines: cx cy R L sev        # storm: center, footprint radius, accumulation limit, severity (per-mille)
N lines: x y e p tech         # candidate i: location, exposure, premium, technical price
```
All values are integers, 0-indexed candidates `0..N-1`. `sev` is the storm's loss rate
in per-mille (e.g. `600` = 0.6 of exposure lost per unit, before the `OVER_MULT`
multiplier applies to the portion above `L`).

## Output (stdout)
```
m
i_1 i_2 ... i_m
```
`m` (the number of policies you write) followed by `m` distinct indices in `[0, N)`.

## Feasibility
`0 <= m <= C`, all indices distinct and in range. Any violation, or an unparsable
artifact, scores `0`.

## Objective
For a written set `S`, let `X_s` = sum of `e_i` over `i in S` inside storm `s`'s
footprint. Score:
```
F(S) = sum_{i in S} (p_i - tech_i)  -  sum_s  sev_s/1000 * ( min(X_s, L_s) + OVER_MULT * max(0, X_s - L_s) )
```
Maximize `F`.

## Scoring
The checker builds its own reference book `B` (a timid underwriter who ranks
candidates by isolated margin `p_i - tech_i` and writes only the top `ceil(C/3)` of
them, leaving the rest of the capacity idle, storms ignored) and reports
`Ratio = min(1000, 100 * F / B) / 1000`, clamped to `[0, 1]`.

### Worked example
One storm: center `(500,500)`, `R=100`, `L=40`, `sev=600`. `OVER_MULT=3`. Candidates:
`0: (500,500) e=30 p=50 tech=20` (margin 30, in footprint), `1: (520,500) e=25 p=45
tech=20` (margin 25, in footprint, distance 20 <= 100), `2: (900,900) e=40 p=45
tech=25` (margin 20, outside the footprint). `C=2`.

- Writing `{0,1}` (the two highest-margin candidates, footprint ignored): `X_0 = 55`,
  over the limit by `15`. Loss `= 0.6*(40 + 3*15) = 51`. `F = (30+25) - 51 = 4`.
- Writing `{0,2}`: `X_0 = 30` (only policy 0 touches the storm), under the limit. Loss
  `= 0.6*30 = 18`. `F = (30+20) - 18 = 32`.

`{0,2}` diversifies away from the crowded footprint and nets 8x more than the
naive top-margin pick `{0,1}` — the same policy 1 that looks like free money in
isolation is what breaks the book once policy 0 already occupies the footprint.

## Constraints
`8 <= N <= 30`, `2 <= C <= N`, `1 <= K <= 8`, coordinates and radii in `[0,1000]`,
`1 <= e_i <= 200`, technical prices and premiums fit in 32-bit signed integers,
`sev in [1,999]`, `OVER_MULT` a small positive integer. Time limit 5s, memory 512MB.
