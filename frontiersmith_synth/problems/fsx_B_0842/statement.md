# Edge Cache: Prefetching Along the Drifting Hot Set

An edge node on a content-delivery network serves a fixed catalog of `M` content
items over `T` discrete time steps from a small local cache holding at most `C`
items at any one step. You are given, up front, the **exact request-weight
table** `weight[t][i]` for every item `i` at every step `t`. Every item has a
small "evergreen" baseline popularity, but on top of that a contiguous block of
catalog ids is additionally **hot** at any given moment, and the position of that
hot block **drifts** across the catalog over time in a fixed, deterministic way
baked into the table (it can wrap around, so ids that cooled off can become hot
again later). You must decide, for **every** time step, which items sit in the
cache — you see the whole table before committing to anything; there is no
online replay.

## Costs

- **Cache hit**: serving item `i`'s traffic at step `t` while it is cached costs
  `hit_cost * weight[t][i]` (`hit_cost` is always `1.0`).
- **Cache miss**: serving item `i`'s traffic at step `t` while it is NOT cached
  costs `miss_cost[i] * weight[t][i]`, with `miss_cost[i]` well above `hit_cost`
  — an uncached hot item is expensive to serve from origin.
- **Fetch**: whenever item `i` is **newly present** in the cache at step `t` (it
  was absent at `t-1`, or `t=0`), you pay a flat `fetch_cost[i]`, whether this is
  item `i`'s first-ever fetch or a repeat after an earlier eviction. Holding an
  already-cached item costs nothing extra per step. This is the core asymmetry:
  eviction is free, but evicting something you will want again soon just buys
  you a second fetch bill on top of however many misses you eat while it is gone.

**One-step fetch latency.** Items placed into the cache at step `t` do **not**
serve step `t`'s own traffic — only whatever was **already resident going into
step `t`** (the cache as it stood after step `t-1`'s changes; empty before step
`0`) decides hit vs. miss for step `t`. Items added during step `t` only start
serving traffic from step `t+1` onward. So the only way to avoid a miss on an
item that turns hot at step `t` is to have already fetched it by the end of step
`t-1` — you must act on tomorrow's row of the table, not today's.

The **total cost** to minimize is the fetch cost of every newly-entering item at
every step, plus the hit/miss cost of every item's traffic (judged against the
cache as it stood *before* that step's own changes), summed over `t = 0..T-1`.

Cache capacity is tight: `C` is only a little above the hot block's width, so you
cannot hoard the whole catalog — you can prefetch a few ids ahead of the drift,
but making room for what is about to become hot requires evicting something, and
eviction timing genuinely changes the total cost.

## Public instance (stdin JSON)

```json
{
  "M": 44, "T": 32, "C": 11,
  "hit_cost": 1.0,
  "miss_cost":  [7.9, 11.3, ...],   // length M, per item
  "fetch_cost": [61.2, 84.0, ...],  // length M, per item
  "weight": [ [w_0_0, w_0_1, ..., w_0_(M-1)],   // step 0
              [w_1_0, w_1_1, ..., w_1_(M-1)],   // step 1
              ... ]                              // T rows total
}
```

## Answer (stdout JSON)

```json
{"cache": [ [ids at step 0], [ids at step 1], ..., [ids at step T-1] ]}
```

Each inner list must have **at most `C`** distinct integer item ids in `[0, M)`
(fewer is allowed). Any malformed answer, wrong length, out-of-range id,
duplicate id within a step, or a step exceeding capacity `C` is **infeasible and
scores 0**.

## Scoring

The evaluator computes a baseline `b` = the cost of the single best **fixed**
cache: one set of ids, chosen once (weighing each item's own `miss_cost` and
`fetch_cost` against how much traffic it draws from step 1 onward) and then held
for every step — the best possible design that ignores the drift entirely. For a
feasible answer with total cost `obj`:

```
r = min(1, 0.1 * b / obj)
```

so the best fixed-cache design maps to exactly `0.1`; a design `k` times cheaper
maps to `min(1, 0.1k)`. `Ratio` is the mean of `r` over 10 deterministic, seeded
instances, including larger held-out ones with faster or repeatedly-wrapping
drift. Your program reads one public instance JSON from stdin and writes one
answer JSON to stdout, in an **isolated subprocess** seeing only that instance.
