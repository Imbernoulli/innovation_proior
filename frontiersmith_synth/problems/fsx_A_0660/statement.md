# Monsoon Cascade: Three-Dam Flood Policy

It is monsoon season on a river with three dams in series: Dam 1 (upstream),
Dam 2 (middle), Dam 3 (downstream) — and a town just below Dam 3. Released
water does not arrive instantly downstream: it travels. Water Dam 1 releases
at tick `t` shows up as extra inflow to Dam 2 at tick `t + delay12`; water
Dam 2 releases shows up at Dam 3 at tick `t + delay23`. Each dam also has its
own local catchment inflow, and the town has its own small local inflow. You
are handed the FULL forecast for the whole event (every hydrograph, every
tick) and must commit, in advance, to a release schedule for all three gates.

**Physics each tick (in dam order 1 → 2 → 3):**

```
avail_i        = storage_i + own_inflow_i(t) + routed_in_i(t)
requested      = clip(your release_i[t], 0, release_max_i)
actual_i(t)    = min(requested, avail_i)          # can't release water you don't have
leftover       = avail_i - actual_i(t)
if leftover > capacity_i:                          # UNCONTROLLED emergency spill
    actual_i(t) += leftover - capacity_i
    storage_i(next) = capacity_i
else:
    storage_i(next) = leftover
```

`routed_in_1 = 0`; `routed_in_2(t) = actual_1(t - delay12)` (0 if `t < delay12`);
`routed_in_3(t) = actual_2(t - delay23)` (0 if `t < delay23`).

**Town flow:** `town_flow(t) = actual_3(t) + town_inflow(t)`.

**Objective (minimize):** the flood peak, `max over t of town_flow(t)`, over a
fixed seeded family of 10 monsoon events (varying horizon length, dam sizes,
gate rate caps, routing delays, and — critically — how the upstream pulse's
arrival time lines up with the dam 3 / town local pulses).

A dam that holds water too long gets caught with no room left when its
inflow finally arrives, forcing a large uncontrolled spill. If that forced
spill coincides with the town's own local flood pulse, the downstream peak
is far worse than either alone — releasing *before* you look full, to create
absorption headroom ahead of a delayed pulse, can beat reacting once full.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from
**stdin**, write ONE JSON object (your answer) to **stdout**. Runs isolated;
sees only the public instance below.

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide a release schedule ...
print(json.dumps({"release1": [...], "release2": [...], "release3": [...]}))
```

### Public instance (stdin)

```json
{
  "name": "cascade901", "t_steps": 40,
  "dam1": {"capacity": 260.0, "storage0": 130.0, "release_max": 26.0, "inflow": [.., ..]},
  "dam2": {"capacity": 220.0, "storage0": 110.0, "release_max": 22.0, "inflow": [.., ..]},
  "dam3": {"capacity": 170.0, "storage0":  85.0, "release_max": 18.0, "inflow": [.., ..]},
  "delay12": 5, "delay23": 4,
  "town_inflow": [.., ..]
}
```
(every `inflow`/`town_inflow` list has length `t_steps`)

### Answer (stdout)

```json
{ "release1": [T floats], "release2": [T floats], "release3": [T floats] }
```

Each list must have **exactly `t_steps`** finite, non-negative numbers.
Requesting more than `release_max` is simply clipped by physics (not a
foul). A **negative** or **non-finite** value, wrong type, or wrong length
makes the *whole instance* score `0.0` — as does a crash, timeout, or
non-JSON output.

## Scoring (deterministic)

For each instance the evaluator computes, itself, never trusting the candidate:

- `q_base` — peak town flow from an internal **reactive per-dam threshold**
  policy (each dam reacts only to its own current storage fraction: full
  throttle above 80% full, a third-throttle above 55%, otherwise closed —
  no forecast, no coordination),
- `q_lb` — an instance-only **lower bound** on any policy's peak:
  `max(max_t town_inflow(t), (storage0_3 + sum(dam3.inflow) - capacity_3) / t_steps)`,
- `q_cand` — the peak town flow your submitted schedule actually produces.

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```

Matching the reactive baseline scores ≈ `0.1`; reaching the (generally
unreachable, loose) lower bound scores `1.0`; doing worse than the baseline
scores below `0.1`. The reported **Ratio** is the mean of `r` over all 10
instances; **Vector** lists the per-instance scores.

## Suggested strategies

1. **Constant rate**: fixed fraction of max rate the whole time — ignores
   everything.
2. **Reactive threshold**: release more as each dam's own storage fills up —
   the obvious first idea, blind to the forecast and the other dams.
3. **Cascade-order lookahead**: plan dam 1 first, feed its (delayed) release
   into dam 2's forecast and plan dam 2, then likewise plan dam 3 — using the
   forecast to release *before* a deadline instead of reacting to it.
4. **Anticipatory peak-shaving**: bias the downstream dam to draw down ahead
   of the town's own forecast pulse, trading a locally wasteful early
   release for a smaller combined downstream peak.
