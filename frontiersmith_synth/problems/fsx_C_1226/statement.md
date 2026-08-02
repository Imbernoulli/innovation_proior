# Pacing the Collector

## Problem

You control a two-generation heap over `T` discrete time steps. At each step `t`
(`1..T`) a batch of `alloc[t]` new objects is allocated into the **young**
generation; every object in that batch shares a fixed lifetime `lifetime[t]`
and dies exactly at time `t + lifetime[t]` (it is garbage from that step on,
but stays occupying memory until a collection actually sweeps it).

For every step you choose one action:
- `0` — do nothing,
- `1` — run a **minor** collection: scan every object currently resident in
  the young generation. Dead ones (death time already reached) are freed.
  Each surviving object's *survive-count* increases by 1; once an object's
  survive-count reaches the promotion threshold `K` it is **tenured** — moved
  permanently into the **old** generation (it will never be scanned by a minor
  collection again),
- `2` — run a **major** collection: does everything a minor collection does,
  *and* also scans the old generation, freeing old objects whose death time
  has been reached.

A collection's **pause length** is `c0 + k_young * (young objects resident
right before the scan)`, plus `k_old * (old objects resident right before the
scan)` for a major collection (using `c0_major` instead of `c0_minor`). Every
step, whether or not you collect, you additionally pay a small **footprint
cost** `f_rate * (young resident + old resident)` — memory you are holding,
scanned or not.

**Feasibility (hard):** the young generation has capacity `Y`. Right after a
step's allocation (and this step's own action, if any, which runs *before*
the allocation), the young-resident count must never exceed `Y`, or the
submission is infeasible (score 0).

**Pause budget (soft, but expensive):** any single collection whose pause
length exceeds `Bpause` is not infeasible, but adds an extra
`penalty * (pause_length - Bpause)` on top of its own pause cost — the
overshoot is punished in proportion to how far over budget it is, so one
enormous, deferred sweep is far worse than several small overshoots of the
same total size. Collect-when-full keeps the *number* of collections low but
tends to build up a huge scan right when the heap is finally forced to
collect, busting the budget badly.

Minimize total cost = sum of all pause lengths + `penalty` times the total
overshoot above `Bpause` (summed over every violating collection) + sum of
all footprint costs.

## Input (stdin)
```
T Y K Bpause c0_minor c0_major k_young k_old f_rate penalty
alloc[1] alloc[2] ... alloc[T]
lifetime[1] lifetime[2] ... lifetime[T]
```
`f_rate` is a decimal; every other value is an integer. `1 <= lifetime[t]`.

## Output (stdout)
Exactly `T` whitespace-separated integers `a[1..T]`, each in `{0,1,2}` — the
action taken at every step, in order.

## Scoring
The checker replays your schedule exactly as specified above. Any feasibility
violation (wrong token count, non-integer/out-of-range token, or a step where
young-resident exceeds `Y`) scores `Ratio: 0.0`. Otherwise let `F` be your
total cost and `Bref` the total cost of the checker's own "minor-collect
every step" construction on the same instance:
`Ratio = min(1.0, 0.1 * Bref / F)`.
Lower cost is better; scores are averaged over 10 hidden test cases.

## Worked example (illustrative, small numbers)
`T=4 Y=10 K=2 Bpause=15 c0_minor=5 c0_major=12 k_young=1 k_old=2 f_rate=0.1
penalty=5`, `alloc=[3,0,4,0]`, `lifetime=[2,1,2,1]`.

Submission `0 1 0 1`: step 1 allocates 3 (resident 3, footprint 0.3). Step 2
minor-collects (pre-scan resident 3, cost `5+3=8`, no violation), then
resident stays 3 (that cohort is still alive, survive-count 1<2). Step 3
allocates 4 more (resident 7, footprint 0.7). Step 4 minor-collects (pre-scan
resident 7, cost `5+7=12`): the age-3 cohort is now dead and freed, the other
survives (survive-count 1<2), leaving resident 4. Total `F = 21.7`.

The checker's baseline "collect every step" (`1 1 1 1`) totals `Bref = 31.4`
on the same instance (more collections, but each is cheap since nothing ever
accumulates). `Ratio = min(1, 0.1*31.4/21.7) = 0.1447`.

## Constraints
`1 <= T <= 200`, `0 <= alloc[t] <= 60`, `1 <= lifetime[t] <= T`,
`Y, Bpause, c0_minor, c0_major, k_young, k_old, penalty` positive integers
within `int32` range, `K` a small positive integer, `0 <= f_rate <= 1`. Time
limit 5s, memory 512MB.
