# Trip Fast, Recover Slow

## Problem
A service calls a flaky downstream dependency once per tick over a fixed
timeline of `T` ticks. Whether a call attempted at tick `t` would have
succeeded is already decided (given in the input) -- your job is to tune a
**circuit breaker** that decides, tick by tick and using only what it has
observed so far, whether to call at all. You submit breaker **parameters**;
a fixed replay engine then walks the timeline tick by tick applying your
parameters, exactly as specified below.

The breaker has three states. **CLOSED**: every tick is called. Track the
outcomes of the last `w_trip` calls; if `k_trip` of them failed, **TRIP** to
OPEN. **OPEN**: no calls are made except single spaced-out probes, one every
`probe_interval` ticks (`probe_interval` is derived from your parameters,
see below). A failed probe changes nothing (keep waiting). A successful
probe moves to **HALF_OPEN**. **HALF_OPEN**: calls resume. Two independent
checks run every tick: (a) the same short-window rule as CLOSED, over the
last `w_trip` HALF_OPEN outcomes -- `k_trip` failures sends it straight back
to OPEN (a *false recovery*); (b) a **separate**, independently-sized window
of the last `w_recover` HALF_OPEN outcomes -- once that window is full and
`k_recover` of them succeeded, the breaker fully **CLOSES**.

`probe_interval` is fixed for the whole duration of one OPEN episode, decided
the moment a FRESH trip happens from CLOSED (a false-recovery re-trip from
HALF_OPEN keeps the episode's existing `probe_interval` unchanged). If
`probe_num=0`, it is always `probe_base`. Otherwise it is
`clamp(round(probe_num/probe_den * G), 1, T)`, where `G` is the *signal gap*
of the most recently fully-CLOSED episode: the number of ticks from that
episode's trip to the first probe that came back successful (or
`probe_base` if no episode has fully closed yet).

## Input (stdin)
```
T R CF
o_1 o_2 ... o_T
```
`o_t in {0,1}`: whether a call at tick `t` would succeed. `R` is the reward
for a successful call, `CF` the cost of a failed call.

## Output (stdout)
Seven integers on one line:
```
w_trip k_trip w_recover k_recover probe_base probe_num probe_den
```

## Feasibility
Reject (score `0`) unless: exactly 7 finite integer tokens; `1<=w_trip<=T`;
`1<=k_trip<=w_trip`; `1<=w_recover<=T`; `1<=k_recover<=w_recover`;
`1<=probe_base<=T`; `1<=probe_den<=1000`; `0<=probe_num<=probe_den`.

## Objective
Replay the timeline with your parameters. Every tick where a call is
actually attempted (CLOSED, HALF_OPEN, or an OPEN-state probe) contributes
`+R` if `o_t=1` or `-CF` if `o_t=0`; a skipped tick (OPEN, not a probe tick)
contributes `0`. Maximize the total `F`.

## Scoring
The checker's own reference is a breaker configured to never trip in
practice (identical to making every call with no protection at all),
computed directly from the input timeline as `B = sum(R if o_t else -CF)`.
```
sc = min(1000, 100*F/max(1e-9,B));  Ratio = sc/1000
```

## Constraints
`20<=T<=520`, `R=1`, `CF=3`. Time limit 5s, memory 512MB.

## Example
`T=10, R=1, CF=3`, outcomes `1 1 1 0 0 0 0 1 1 1`. Submit
`w_trip=3 k_trip=2 w_recover=2 k_recover=2 probe_base=3 probe_num=0
probe_den=1` (fixed `probe_interval=3`, never adapted). Ticks 1-3 CLOSED, all
succeed (`F=1,2,3`). Tick 4 fails (`F=0`); tick 5 fails: of the last 3 calls
(ticks 3-5 = `1,0,0`), 2 failed -> `k_trip=2` reached -> TRIP, OPEN,
`F=-3`, `next_probe=5+3=8`. Ticks 6-7 skipped (no cost, no reward). Tick 8
probes: succeeds (`F=-2`) -> HALF_OPEN. Tick 9: succeeds (`F=-1`); the
`w_recover=2` window is now full with 2 successes -> `k_recover=2` reached
-> fully CLOSES. Tick 10: CLOSED again, succeeds (`F=0`). Total `F=0`.
Baseline `B = 6*1 - 4*3 = -6 -> max(1e-9,-6)=1e-9`, so `Ratio=0.0` here --
this shape (naive-baseline going non-positive) never happens on the real 10
scored cases, whose timelines are dominated by healthy stretches by
construction. This tiny walkthrough is shown only to make the tick-by-tick
mechanics concrete; the scored cases are larger and interleave several
outage/gray-failure stretches, some with very different recovery durations
from each other and some that are genuinely noisy (~50% success) rather than
a clean up/down transition -- exactly the regime where a single symmetric
window/threshold with a fixed probe cadence tends to flap.
