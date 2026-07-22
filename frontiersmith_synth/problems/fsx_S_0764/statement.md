# Two Strata of Cold: A Polar Station's Storage Portfolio

You schedule power for an off-grid polar research station over `T` discrete ticks.
Each tick has a public renewable generation `gen_t` (wind) and a public station load
`load_t`. There is **no grid**: whatever the station consumes must come from
generation or storage *that tick*.

## Two stores, two jobs

- **Battery**: small capacity, high round-trip efficiency, fast — the everyday
  buffer for ordinary hour-to-hour swings.
- **Fuel store** (electrolysis → synthetic fuel → fuel cell): huge capacity, but a
  **lossy** round trip — built to ride out multi-tick generation **droughts** that
  the small battery physically cannot hold enough energy to cover.

Charging a store draws bus power (only an `eta_in` fraction is actually stored);
discharging draws stored energy (only an `eta_out` fraction reaches the bus). Both
are rate-limited per tick (`rate`) and capacity-limited in total (`cap`).

A handful of instances contain planted multi-tick droughts: `gen` collapses far
below `load` for many consecutive ticks. The battery alone cannot hold enough energy
to bridge one. The fuel store can — but only if it has been given a long, early lead
to accumulate stored energy, because its round trip is far worse than the battery's.

## Input (one JSON object on stdin — the public instance)

```
{"name": str, "T": int,
 "load": [load_0 … load_{T-1}], "gen": [gen_0 … gen_{T-1}],
 "battery": {"cap": f, "rate": f, "eta_in": f, "eta_out": f},
 "fuel":    {"cap": f, "rate": f, "eta_in": f, "eta_out": f},
 "blackout_coef": f}
```

The entire trace is public, so any drought is visible in advance if you look for it.

## Output (one JSON object on stdout)

```
{"bc": [ … T … ], "bd": [ … T … ],   # battery charge / discharge request, each >= 0
 "fc": [ … T … ], "fd": [ … T … ]}   # fuel charge / discharge request,    each >= 0
```

Each list must have exactly `T` finite, non-negative numbers. Any violation, crash,
timeout, or non-JSON scores 0 on that instance. Requests are **clamped** by the
evaluator to rate limits and to physically available power / stored energy — an
infeasible *request* is honestly resolved, not rejected outright.

## Balance rule (evaluated by the scorer, not you)

Each tick, in order: (1) discharges are clamped to rate and to what is stored;
delivered power = `eta_out · discharge`; (2) available = `gen_t + delivered`;
(3) **load is served first**, up to available (shortfall is "unserved"); (4) power
left over may charge a store — battery first, then fuel, each clamped to rate and
remaining room; anything still left is curtailed (lost, no separate penalty).

## Objective (maximize)

```
obj = sum_t served_t  -  blackout_coef * sum_t unserved_t
```
where `served_t + unserved_t == load_t` every tick.

## Scoring (deterministic; no wall-time)

For each of 10 fixed seeded instances, let `b` be the objective of submitting
all-zero vectors (no storage action — generation serves load directly whenever it
can) and `hi` a generous, **unreachable** upper bound (load served in full minus
whatever deficit would remain even with both stores pre-charged and discharging at
maximum simultaneous rate the whole time — ignoring provisioning-time limits
entirely):

```
r = clamp( 0.1 + 0.9 · (obj - b) / (hi - b),  0, 1 )
```

Doing nothing scores exactly `0.1`. Your score is the **mean of `r`** over the 10
instances: calm (no drought) traces, single-drought traps of varying
length/severity/lead-time, and held-out twin-drought (two droughts in one trace)
generalization instances.

## What to notice

A converter idle 80-90% of the time, losing ~40% of whatever passes through it,
*looks* like waste by any efficiency metric — so the obvious recipe cycles only the
small, efficient battery and skips the lossy store. Fine for routine noise, but the
battery's tiny capacity drains in a couple of ticks against a real drought. The two
stores hedge *different* probability strata: only a shortfall that genuinely exceeds
what the battery could ever cover is worth the fuel store's poor round trip — and
only if you bank surplus into it long before the drought arrives, since it needs far
more raw energy, over far more ticks, to store the same amount you get back out.
