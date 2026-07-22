# Depleting the Budget: Retrofit Scheduling Against a Demand Calendar

## Problem

You operate a fleet of `U` power units over a horizon of `T` steps. Unit `i`
has capacity `C_i` (MW, unchanged for its whole life), a dirty emission rate
`R_i` (tons/MWh) it emits at until retrofitted, a duration `D_i` (steps) for
its retrofit, and a clean emission rate `r_i < R_i` it emits at afterward.
You may retrofit each unit **at most once**: choose a start step `s_i`
(`0` = never retrofit). While retrofitting (`s_i <= t <= s_i + D_i - 1`) the
unit is **fully offline** (zero capacity). Before `s_i` it runs at `(C_i,
R_i)`; from `s_i + D_i` onward it runs at `(C_i, r_i)`.

A demand series `demand[t]` for `t = 1..T` must be met **every step** by the
combined dispatch of all online units (a fleet-coverage constraint) — you
may also dispatch *more* than demand for extra credit (surplus energy
sold elsewhere), up to each online unit's capacity.

There is a single cumulative emissions **budget** `BUDGET` for the whole
horizon. Let `cum_em` be the sum, over every step and unit, of
`dispatch(i,t) * rate(i,t)`. If `cum_em` exceeds `BUDGET`, you are charged a
penalty of `PEN * max(0, cum_em - BUDGET)` (`PEN` given in the input) against
your total energy served — the budget behaves like a store of "clean room"
that only depletes, never refills, and every unit of overage is charged the
same rate regardless of when it happened. Your objective is:

```
F = max(0, total_energy_served - PEN * max(0, cum_em - BUDGET))
```

Maximize `F`. Note that emitting heavily early (while everything is still
dirty) leaves less headroom later, and that an offline window timed against
a demand trough costs far less coverage-forcing dirty dispatch than one
timed against a peak.

## Input (stdin)

```
U T
PEN BUDGET
C_1 R_1 r_1 D_1
...
C_U R_U r_U D_U
demand_1 demand_2 ... demand_T
```
`PEN`, `BUDGET`, `C_i`, `R_i`, `r_i`, `demand_t` are real numbers; `D_i` is a
positive integer. `R_i > r_i > 0` for all `i`.

## Output (stdout)

```
s_1 s_2 ... s_U
d_{1,1} d_{2,1} ... d_{U,1}
...
d_{1,T} d_{2,T} ... d_{U,T}
```
`s_i` is unit `i`'s retrofit start (`0` or an integer in `[1, T-D_i+1]`).
Row `t` gives unit `i`'s dispatch `d_{i,t} >= 0` at step `t`.

## Feasibility

Every `s_i` must be `0` or within `[1, T - D_i + 1]`. Every `d_{i,t}` must be
a finite, non-negative number not exceeding unit `i`'s capacity at step `t`
(`0` while offline, `C_i` otherwise). At every step `t`, `sum_i d_{i,t}` must
be `>= demand_t`. Any violation, wrong token count, or non-numeric/
non-finite token makes the output infeasible (`Ratio: 0.0`).

## Scoring

The checker replays your schedule and dispatch to get `F` as above, and
compares it to `B`, the same quantity for a fixed naive baseline: never
retrofit anyone, and fill demand each step in **input index order** up to
each unit's capacity (no bonus dispatch). Final score:
`min(1, F / (10*B))`.

*Illustrative example only* (not a real test case): with `U=2`, a short
horizon, and a demand calendar with a trough right at the start, retrofitting
the dirtier unit during that trough (so no coverage strain occurs) and then
selling clean surplus energy for the rest of the horizon gives high `F` at
low `cum_em` — a strong, feasible schedule.

## Constraints

`5 <= U <= 9`, `30 <= T <= 70`, `C_i, R_i, r_i, demand_t > 0`, `D_i >= 3`,
`PEN >= 0`, `BUDGET > 0`. Time limit 5s.
