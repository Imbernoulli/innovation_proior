# Wear Before the Rush: Fatigue-Budgeted Fleet Routing

You control a small fleet of machines that must process a stream of incoming
jobs over a fixed number of time ticks. Every job you route costs the unit
that handles it a little bit of **fatigue**; too much cumulative fatigue and
the unit **fails** and needs repair. Your goal is to keep the fleet's overall
throughput high across the whole horizon -- including through one seeded,
denser demand **surge** somewhere in the schedule.

## Setup

For each of 10 hidden test instances you get the **entire** instance up front
(this is an offline routing problem -- nothing about the future is hidden from
you):

```
{"n_units": N, "horizon": T,
 "units": [{"id":.., "capacity":.., "fatigue_rate":.., "hazard_cliff":..,
            "repair_time":.., "recover_rate":..}, ...],
 "jobs":  [{"id":.., "t":.., "weight":1.0}, ...]}
```

`jobs` is sorted by arrival tick `t` (0-indexed, `t < T`); several jobs can
share the same tick. Every job has `weight = 1.0` (the objective is pure
throughput). Each unit has:
- `capacity`: the maximum number of jobs it can **start** in a single tick.
- `fatigue_rate`: fatigue added to the unit's cumulative fatigue for every job
  it processes.
- `hazard_cliff`: the fatigue threshold. The instant cumulative fatigue
  reaches (or crosses) this value, the unit **fails**.
- `repair_time`: on failure, the unit is completely unavailable (capacity 0)
  for the next `repair_time` ticks; once repair ends, its fatigue resets to 0.
- `recover_rate`: on any tick where the unit is NOT in repair and is given
  **zero** jobs, its fatigue decreases by this amount (floored at 0) -- rest
  banks fatigue headroom back.

## What you submit

**Answer** (stdout): `{"assignment": [u_0, u_1, ..., u_{m-1}]}` -- one integer
unit id (`0 <= u_i < N`) for every job, **in the same order as the input
`jobs` array**. `m` is the number of jobs in that instance.

## Simulation (how your assignment is scored)

The evaluator replays the horizon tick by tick, `t = 0 .. T-1`:
1. For every unit not currently under repair that receives **zero** jobs this
   tick, its fatigue decays by `recover_rate` (floored at 0).
2. Jobs arriving at tick `t` are processed in the order they appear in the
   `jobs` array. For a job routed to unit `u`:
   - If `u` is still under repair at tick `t` (i.e. within `repair_time`
     ticks of its last failure), the job is **dropped** (0 value).
   - If `u` has already started `capacity` jobs this tick, the job is
     **dropped** (0 value) -- capacity is a hard per-tick limit, not a queue.
   - Otherwise the job is completed: its weight is added to your score, and
     `fatigue_rate` is added to `u`'s cumulative fatigue. If that pushes
     fatigue to or past `hazard_cliff`, `u` fails immediately: it becomes
     unavailable for the next `repair_time` ticks and its fatigue is reset to
     0 once repair ends.

An instance's score is `completed_weight / total_weight` (both sums over that
instance's jobs), a value in `[0, 1]`. **Every instance is deliberately
over-subscribed**: total demand exceeds what the fleet can sustain even under
excellent routing, so a perfect score is not attainable by design -- there is
always headroom above the best strategy you can build. A malformed answer
(wrong type, wrong length, an out-of-range or non-integer unit id) scores 0 on
that instance. Your final `Ratio` is the mean of the 10 per-instance scores.

## Why the obvious approach fails

A natural first policy is to always route each job to whichever available
unit is fastest (largest `capacity`) right now -- maximize throughput this
instant. But in these fleets capacity and fatigue_rate are correlated: the
fast unit is also the one that wears fastest. Chasing instantaneous
throughput drives that unit's cumulative fatigue up continuously through the
quiet part of the schedule, so by the time the seeded surge arrives, it is
already worn down or mid-repair -- exactly when its extra capacity is needed
most, and exactly when losing it is most expensive. Balancing *cumulative*
fatigue over the whole horizon, not *instantaneous* load in the current tick,
is what keeps every unit's hazard cliff out of reach when the surge hits.

## Constraints

Time limit 2s per candidate call, memory 512MB. Each instance has at most a
few hundred jobs. Objective: **maximize** the mean per-instance throughput
ratio.
