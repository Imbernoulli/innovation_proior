# Sleeper-Train Overbooking: One Ladder, Ninety Nights

## Problem

You run the berths on a sleeper train for a season of `N` consecutive nights. Night `i` has
`capacity_i` physical berths and a fare `fare_i` charged for every berth **sold**, whether or
not the passenger actually shows up. You may sell up to `max_sell_i` berths that night
(`max_sell_i >= capacity_i`; the gap is the night's printed overbooking headroom).

For night `i`, the input prints, in sale order `j = 1..max_sell_i`, every prospective
passenger's fixed **no-show bit** (`1` = they will not show) and **volunteer threshold**
(the compensation at which they would voluntarily give up their berth if bumped). All of this
is decided in advance and printed to you — nothing is probabilistic at scoring time.

You choose, for each night, how many berths to actually sell: `sold_i` with
`0 <= sold_i <= max_sell_i`. The passengers who show up are exactly the first `sold_i`
printed passengers whose no-show bit is `0`. If more passengers show than there are berths
(`shows_i > capacity_i`), the excess `overflow_i = shows_i - capacity_i` must be removed.

You also choose **one global** compensation ladder: five strictly increasing integers
`s_1 < s_2 < s_3 < s_4 < s_5` (bounds `LADDER_LO`, `LADDER_HI` given in the input), shared by
every night of the season. Removal is resolved by a fixed, deterministic rule: each shown
passenger `p` with threshold `t_p` would volunteer at the **cheapest ladder step that covers
them**, `cost_p = min{ s_k : s_k >= t_p }` (or `cost_p = infinity` if `t_p > s_5` — no step
covers them). Among the `shows_i` shown passengers, the `overflow_i` with the **smallest**
`cost_p` (ties broken by passenger index) are the ones removed: those with finite `cost_p`
are paid it (volunteers), the rest are bumped involuntarily and each such bump costs a fixed
penalty `PENALTY` (also given in the input) — no compensation is paid for them.

Every 5th night is a high-fare "peak" night. Peak nights tend to have noticeably lower
no-show rates and noticeably higher volunteer thresholds than ordinary nights — a night's
fare, its no-show rate, and its threshold level are correlated, and that correlation is not
uniform across the season.

## Input (stdin)
```
N
LADDER_LO LADDER_HI PENALTY
capacity_1 fare_1 max_sell_1
noshow_1 threshold_1 noshow_2 threshold_2 ... noshow_{max_sell_1} threshold_{max_sell_1}
... (repeated for each of the N nights)
```

## Output (stdout)
```
s_1 s_2 s_3 s_4 s_5
sold_1 sold_2 ... sold_N
```
All values integers. The ladder must satisfy `LADDER_LO <= s_1 < s_2 < ... < s_5 <= LADDER_HI`.
Each `sold_i` must satisfy `0 <= sold_i <= max_sell_i`.

## Feasibility
Wrong token count, non-integers, an out-of-range or non-increasing ladder, or any
`sold_i` outside `[0, max_sell_i]` scores `0`.

## Objective
Maximize total season net: for each night, `sold_i * fare_i` minus compensation paid to
volunteers minus `PENALTY` times the number of involuntary bumps, summed over all `N` nights.

## Scoring
The checker replays your ladder and sale caps exactly as specified above to get your net
`F`, compares against its own conservative reference net `B` (selling a small fixed fraction
of each night's capacity, never overbooking, so it never triggers a single bump), and reports
`Ratio = min(1000, 100*F/B) / 1000`.

## Example (worked, illustrative shape only)
One night: `capacity=10, fare=100, max_sell=12`. Selling `sold=12`: if 11 show
(`overflow=1`) and the cheapest coverable shown passenger has `cost_p=40`, net =
`12*100 - 40 = 1160`, versus `sold=10` (no risk) giving flat `1000`. Whether the riskier cap
pays off depends entirely on how many will show and how cheaply the ladder covers them — read
the printed data, don't assume an average.

## Constraints
`1 <= N <= 90`, `30 <= capacity_i <= 60`, `capacity_i <= max_sell_i <= ~1.3*capacity_i`,
`1 <= fare_i <= 600`, `10 <= LADDER_LO`, `LADDER_HI <= 650`, thresholds `>= 1`. Time limit 5s,
memory 512MB.
