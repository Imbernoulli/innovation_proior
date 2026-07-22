# Harvest Job Board: Posted Picking Rates

## Problem

A harvest job board lists `N` fields. Field `i` has value `v_i`, difficulty
`d_i` (1..10), and a deadline `t_i`: it must be picked by the `t_i`-th worker
arrival or its value is lost forever. `M` workers arrive **in a fixed,
printed order** (worker `j` is the `j`-th to show up). Worker `j` has skill
`s_j` (1..20) and a reservation wage `w_j` — the minimum margin they need to
bother working at all.

You post **one rate per field** before anyone arrives — a fixed menu,
never changed afterward. When worker `j` arrives, they look at every field
still available (not yet taken, deadline not yet passed) and compute, for
each, `margin = rate_i - cost(i, j)`, where

```
cost(i, j) = ceil(C_UNIT * d_i / s_j)
```

(higher skill makes hard fields cheaper for that worker — a comparative
advantage). The worker takes the field with the **largest margin** (ties
broken by lowest field index), provided that margin is `>= w_j`; otherwise
they take nothing and leave for good. A taken field is removed from the
pool. This repeats for `j = 1..M`.

Your payoff is `F = H - W`: the total value of fields taken (all taken
fields are automatically on time, since expired ones are removed before
each arrival looks) minus the total rate paid on every field taken. Workers
act purely in their own interest — they don't know or care what is good
for you.

## Input (stdin)

```
N M C_UNIT RMAX
v_1 d_1 t_1        (N lines: field value, difficulty in [1,10], deadline in [1,M])
...
v_N d_N t_N
s_1 w_1            (M lines: worker skill in [1,20], reservation wage -- ARRIVAL ORDER)
...
s_M w_M
```
`30 <= N <= 200`, `M` roughly `2-2.3x N`, `C_UNIT = 20`, `RMAX = 300`.

## Output (stdout)

Exactly `N` integers `r_1 ... r_N`, the posted rate for each field (any
whitespace layout).

## Feasibility

Exactly `N` integer tokens; every `r_i` in `[0, RMAX]`. Any violation
(wrong count, non-integer token, out-of-range value) scores `0`.

## Objective & Scoring

Maximize `F = H - W` from the replay above. The checker also replays the
identical arrival process with every field posted at the **same flat rate**
`R_FLAT = round(0.5 * RMAX)`, giving baseline `B`. Your score is

```
Ratio = max(0.0, min(1.0, 0.1 * F / B))
```

so the flat-rate baseline scores `~0.1` and you must route the market far
better to climb. Posting rates that are needlessly high can make `F`
negative (you can pay more in wages than the value you harvest); the score
is clamped at `0.0` rather than going negative.

## What makes it hard

Paying more for harder fields sounds sensible, but a rate keyed only to a
field's own attributes is blind to **who is arriving when**. If a rate is
generous enough to guarantee *someone* takes a hard field, it is usually
generous enough that the very first eligible arrival — however poorly
suited — snaps it up, both overpaying and burning that field's slot before
a much cheaper specialist would have shown up. The real lever isn't "the
wage level," it's **which worker ends up on which field** — an assignment
problem where deadlines restrict who is even eligible for a field, and rate
differentials are the only tool you have to steer the sequential,
self-interested arrival process toward a good assignment instead of a
first-come one.

## Example scoring

`C_UNIT = 20`. Two fields: field 1 `(v=200, d=5)`, field 2 `(v=150, d=2)`,
both deadline `3`. Three workers arrive: `(skill=5, wage=0)`,
`(skill=10, wage=0)`, `(skill=2, wage=0)`. Post rates `(15, 10)`.

Worker 1: `cost(1,1)=ceil(100/5)=20` → margin `-5`; `cost(2,1)=ceil(40/5)=8`
→ margin `2`. Takes field 2 (`H=150, W=10`).
Worker 2: only field 1 left, `cost(1,2)=ceil(100/10)=10` → margin `5 >= 0`.
Takes field 1 (`H=350, W=25`).
Worker 3: nothing left.

`F = 350 - 25 = 325`.

## Constraints

Time limit 5s, memory 512MB. Scoring is deterministic.
