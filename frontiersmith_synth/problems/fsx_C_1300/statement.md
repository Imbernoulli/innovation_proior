# Locking Down Without Locking Down

An outbreak has been spreading, **uncontrolled**, for 21 days. You take charge
on day 21 and must set an intervention **level** (0=none, 1=light, 2=moderate,
3=strict) for each of the next `future_days` days. Each level cuts
transmission by a fixed multiplier and costs money every day it's active.
Two things make "pick strict and hold it" the wrong answer:

**Delay.** You never see true new infections. You see two derived series over
the 21-day history: `reported_cases` (incubation + testing + reporting delay
`d_rep`, e.g. 7-10 days) and `leading_indicator` (a much shorter delay
`d_lead`, e.g. 2 days — an early-warning proxy such as wastewater
surveillance), both with bounded multiplicative noise. Because
`reported_cases` cannot show you the last `d_rep` days at all, extrapolating
from it alone systematically **underestimates** how far a fast-growing
outbreak has already progressed *right now*.

**Fatigue.** Compliance with an active (level > 0) restriction decays the
longer it's held continuously, eroding its real transmission-reduction
toward zero; a level-0 rest day lets compliance recover. Holding max
restriction for weeks costs the most money AND, by the end, barely works —
pulsing (bursts of restriction separated by rest) buys back compliance for
less cumulative cost.

## Public instance (stdin)

```json
{
  "name": "wave-a1",
  "future_days": 45,
  "d_rep": 7, "d_lead": 2,
  "levels": [{"m":1.00,"cost":0}, {"m":0.80,"cost":6},
             {"m":0.55,"cost":18}, {"m":0.30,"cost":45}],
  "fatigue": {"decay":0.06, "recover":0.10, "floor":0.35},
  "hospital_capacity": 6000, "overflow_penalty": 4.0, "health_weight": 1.0,
  "reported_cases": [0, 0, 0, 0, 0, 0, 0, 41, 44, ...],
  "leading_indicator": [0, 0, 63, 71, 85, ...]
}
```
`levels[k]` gives level `k`'s transmission multiplier `m` (fraction of
baseline transmission retained under full compliance) and its flat daily
`cost`. Both data series have length 21 (day 0..20); entries before a
series's delay has elapsed are 0 (no data yet).

## Answer (stdout)

```json
{ "levels": [1, 1, 3, 3, 3, 0, 0, 1, 2, ...] }
```
Exactly `future_days` integers, each in `[0, len(levels)-1]`, one per future
day (day 21, 22, ...). Malformed output (wrong length, non-integer or
out-of-range level, a crash, a timeout, non-JSON) scores 0 on that instance.

## How your schedule is scored

The evaluator replays the **true**, hidden outbreak trajectory forward under
your schedule. Each day: fatigue updates (compliance drops toward `floor` on
consecutive active days at rate `decay`, recovers at rate `recover` on a
level-0 day); the level's effective multiplier is `1 - compliance*(1-m)`
(fatigue pulls it toward 1, i.e. toward no effect); new infections grow by
that day's effective multiplier times the true growth rate, times remaining
susceptible fraction (self-limiting as the outbreak burns through the
population). Each day adds `health_weight * infections +
overflow_penalty * max(0, infections - hospital_capacity)` to a health cost,
and that day's `cost` to an economic cost. Your objective is
`-(health_cost + economic_cost)` (higher/less-negative is better).

The evaluator also computes, itself, two references from the same true
trajectory: `obj_base` (the objective of "always level 1", ignoring all
data) and `obj_perfect` (a local-search schedule computed with the *true*
growth rate — strictly more information than you have). Your instance score
is
```
r = clamp(0.1 + 0.9 * (your_obj - obj_base) / (1.5 * (obj_perfect - obj_base)), 0, 1)
```
Matching "always level 1" scores ~0.1. The 1.5x stretch means even the
perfect-information reference does not reach 1.0 — there is real headroom
above any schedule built from noisy, delayed data alone. `Ratio` is the mean
over 10 instances (7 fast-growth, 3 slower, 3 with harsher structural
parameters as held-out generalization cases).

## Constraints

Time limit 3s per candidate call, memory 512MB. `future_days` is at most 50.
Objective: **maximize** the mean per-instance ratio.
