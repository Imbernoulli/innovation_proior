# One Shopkeeper Brain: Reorder Policies Across Demand Families

## Story

You run a shop with six product lines. Each line's weekly demand comes from a
different, unlabeled family -- steady-with-season, bursty-with-rare-spikes,
steadily trending, mostly-zero-with-occasional-lumps, plain noisy, or something
that has looked calm so far but might not stay that way. You are shown 30 weeks of
each line's realized demand (never the family's name) and must commit, **once**,
to a reorder policy for the next 70 weeks. There is no second chance to look at how
week 31 actually turned out before deciding week 32's policy: your program is
called a single time and must hand back one formula per line that the shop will
follow mechanically for the rest of the horizon.

## Reorder mechanics (evaluator-executed)

For line `i` at future week `t` (`t = 30..99`, `j = t-30`, `phase = t mod period`):
```
S(t)   = clamp( level[phase] + trend*j + react*(prev_demand - hist_mean), 0, 1e9 )
order  = max(0, round(S(t) - on_hand))
```
`prev_demand` is the *realized* demand of week `t-1` (already known -- no
lookahead), and `hist_mean` is the mean of the 30-week history you were shown.
Stock then updates: `on_hand += order`; `sales = min(on_hand, demand_t)`;
`lost = demand_t - sales`; `on_hand -= sales` (carried to `t+1`); and
```
profit_t = price*sales - holding_cost*on_hand - stockout_cost*lost - unit_cost*order
```
Your goal is to **maximize total profit summed over all 70 future weeks, for all
six lines, on all instances.**

## Input (public instance, one JSON object on stdin)

```json
{"name": "shop03", "period": 12, "history_weeks": 30, "future_weeks": 70,
 "traces": [
   {"trace_id": 0, "history": [ ...30 non-negative ints... ],
    "initial_on_hand": 27, "price": 9.6, "unit_cost": 4.15,
    "holding_cost": 0.45, "stockout_cost": 6.7},
   ... exactly 6 entries, trace_id 0..5 ...
 ]}
```

## Output (one JSON object on stdout)

```json
{"policies": [
   {"trace_id": 0, "level": [ ...12 floats... ], "trend": 0.0, "react": 0.0},
   ... exactly 6 entries, trace_ids forming the set {0,...,5} ...
]}
```
`level` must have exactly `period` (12) finite numeric entries; `trend` and
`react` must be finite numbers. Any missing/duplicate `trace_id`, wrong-length
`level`, non-finite value, or malformed/non-JSON output scores that instance
`0.0`.

## Scoring (deterministic, no wall-time)

For each line the evaluator computes, itself, two references never sent to your
program: `profit_base` from the flat "order-up-to the historical mean, no
adaptation" policy, and `profit_ub`, a loose clairvoyant bound (`sum (price -
unit_cost) * demand_t`, i.e. zero holding/stockout cost and an order that exactly
matches every week's demand -- unreachable by any policy that cannot see the
future, so a strong policy still has headroom below it). Your line's normalized
score is
```
r = clamp( 0.1 + 0.9*(profit_cand - profit_base) / max(1, profit_ub - profit_base), 0, 1 )
```
Matching the flat baseline scores ~0.1 on that line; doing worse scores below
0.1 (floored at 0); adapting to the line's real shape scores higher.

**Each instance's score is the GEOMETRIC MEAN of its six lines' `r` values, not
the average.** A policy that thrives on five lines but lets even one line collapse
toward 0 is punished far more harshly than an arithmetic average would punish it --
under a geometric mean, no line may be sacrificed. Your final score is the mean of
this geometric mean over 10 instances (some noticeably harder than others).

## Notes

- Costs and starting stock differ per line and per instance -- read them, they
  are part of the newsvendor tradeoff (a line where stockouts are far pricier
  than holding inventory wants a much bigger buffer than one where they are not).
- The six `trace_id`s within one instance are always the same six family
  archetypes, but their statistics (magnitude, noise, drift) vary per instance --
  you must detect the family from the numbers, not memorize which slot it sits in.
- Your program is run in an isolated subprocess and sees only the public instance
  above; scoring never depends on wall-clock time.
