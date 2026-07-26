# Island Microgrid: The Twenty-Four Hourly Rates

## Problem

You set the electricity tariff for an island microgrid for the next 24
hours: one price per hour, `p[0..23]`. `N` households share the grid. Each
household `i` has a fixed background load `L[i][t]` for every hour `t`
(cooking, lighting, appliances -- not price-sensitive) and a home
battery/EV charging session that must draw `need[i]` energy units over the
day, at up to `rate[i]` units per hour.

Every household runs the **same published, deterministic** arbitrage
algorithm against whatever tariff you post: it ranks the 24 hours by
`p[t] + eps(i, t)`, ascending, where `eps(i, t)` is a small,
publicly-known, per-household/per-hour tie-break

```
eps(i, t) = ((L[i][t] * 37 + i * 101 + t * 7) mod EPS_MOD) / 10000.0
```

(so `eps(i, t)` lies in `[0, EPS_MOD/10000)`). It then fills its `need[i]`
starting from the top of that ranking, `rate[i]` units per visited hour,
until fully charged. Because the whole fleet runs the *same* algorithm,
your price vector is effectively a scheduling signal for every household
at once -- and the tie-break noise is the only thing that can keep them
from all agreeing on it.

Grid load at hour `t` is `G[t] = sum_i (L[i][t] + charge_i[t])`.

## Input (stdin)

```
N
P_MIN P_MAX ALPHA TOL_FRAC EPS_MOD
L[0][0] L[0][1] ... L[0][23] need[0] rate[0]
...
L[N-1][0] ... L[N-1][23] need[N-1] rate[N-1]
```
All `L`, `need`, `rate` are positive integers; `need[i] > rate[i]`.

## Output (stdout)

Exactly 24 numbers `p[0] ... p[23]`, each within `[P_MIN, P_MAX]`.

## Feasibility

1. Exactly 24 finite numbers, each in `[P_MIN, P_MAX]` (tolerance `1e-6`).
2. **Revenue neutrality.** Let `P0 = (P_MIN+P_MAX)/2`, let `G0` be the
   grid load if you had posted the flat tariff `P0` everywhere, and
   `R0 = P0 * sum(G0)`. Your posted tariff's actual revenue
   `R = sum_t p[t]*G[t]` (with `G` from replaying *your* tariff) must
   satisfy `|R - R0| <= TOL_FRAC * R0`. This is the utility's real budget
   constraint: you may reshape *when* people pay, not raise or lower the
   total take.
Any violation scores `Ratio: 0.0`.

## Objective (minimize)

`F = max_t G[t] + ALPHA * sum_t G[t]^2` -- the worst-hour grid peak, plus a
quadratic generation-cost penalty that rewards a smooth load curve over the
whole day, not just a low single peak.

## Scoring

The checker builds its own reference `G_nr`: the load if nobody responded
to price at all (every household just charges starting at hour 0, in
calendar order). Let `B = max_t G_nr[t] + ALPHA * sum_t G_nr[t]^2`. Then

```
Ratio = min(1000, 100 * B / F) / 1000
```

Lower `F` (relative to `B`) scores higher; 10x better than the do-nothing
reference caps the score at 1.0. Your total score is the mean `Ratio`
over 10 hidden test cases (small to large fleets).

## Worked example (illustrative only, not test data)

`N=2`, `EPS_MOD=100`. Household 0: `L=[5,1]`, `need=3, rate=2`. Household
1: `L=[1,5]`, `need=3, rate=2`. Two hours only (illustration; real cases
have 24). If you post `p=[0.10,0.10]` (flat): `eps(0,0)=(5*37)%100/1e4=
0.0085`, `eps(0,1)=(1*37+7)%100/1e4=0.0044`, so household 0 charges hour 1
first (2 units), then hour 0 (1 unit). Symmetric reasoning sends household
1 mostly to hour 0. Loads spread out instead of both piling on one hour --
this is the desynchronizing effect the tie-break noise buys you, at any
flat or near-flat price level.

## Constraints

`10 <= N <= 260`, `24` hours fixed, integers as generated, time limit 5s,
memory 512MB.
