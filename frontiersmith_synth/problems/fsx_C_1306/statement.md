# Water You Release Now Is Water You Lack Later

A single reservoir serves a downstream demand schedule over a fixed horizon
of `T` days. Every day you choose a **release**. Water arrives as **inflow**
(mostly steady baseflow, occasionally a big pulse). The reservoir has a hard
**capacity**: inflow that pushes storage above capacity, net of that day's
release, **spills** and causes flood damage (super-linear in spill size — a
big spill is disproportionately worse than two small ones). Storage must
never drop below a **dead storage** floor, and release is capped at a
maximum daily rate (you cannot empty a full reservoir in one day).

Each instance ships one long-range inflow **forecast** for the whole
horizon, plus a **skill** curve `skill[t] in [0,1]`: how trustworthy that
forecast is at lead time `t`. Forecasts blend true inflow with the public
seasonal **climatology** baseline — `skill[t]` toward the truth,
`(1-skill[t])` toward climatology. Skill starts near 1 and decays toward a
low floor; the decay rate differs between sites — some see trouble coming
from far away, others ("flash" sites) lose reliable warning within days.
Where skill is low, a big pulse is invisible in the forecast: it just looks
like the ordinary seasonal baseline.

Keeping the reservoir topped up maximizes day-to-day supply security but
leaves no headroom to absorb a pulse you didn't see coming. The right
standing buffer size is a genuine trade-off — and should depend on how far
ahead you can actually trust your forecast, not be a fixed number.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from
**stdin**, write ONE JSON object (your answer) to **stdout**.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a release plan ...
print(json.dumps({"release": release}))
```

### Public instance (stdin)

```json
{
  "name": "trap_fast_late", "T": 50,
  "capacity": 1000.0, "dead_storage": 50.0, "init_storage": 550.0,
  "r_max": 50.0, "flood_cost_coef": 0.05, "flash_risk": 0.95,
  "climatology": [/* T floats: public seasonal baseline inflow */],
  "skill":       [/* T floats in [0,1]: forecast trust at lead t */],
  "forecast":    [/* T floats: skill[t]*true + (1-skill[t])*climatology[t] */],
  "demand":      [/* T floats: downstream demand on day t */]
}
```

`flash_risk` is a site rating: the worst known surge's TOTAL VOLUME as a
fraction of capacity (how big trouble can get, not when — a multi-day
pulse dumps more water than any single day's rate suggests, so size any
standing buffer off this volume, not a rate).

### Answer (stdout)

```json
{ "release": [x_0, x_1, ..., x_{T-1}] }
```

`release` must be a list of exactly `T` finite, non-negative numbers. Any
shape/type violation, a crash, a timeout, or non-JSON output scores that
instance `0.0`.

## Dynamics (evaluator-side, uses the true hidden inflow you never see)

Per day `t`: `applied = clip(release_t, 0, r_max, S + inflow_t - dead_storage)`;
`raw_after = S + inflow_t - applied`; `spill = max(0, raw_after - capacity)`;
`S` updates to `clip(raw_after - spill, dead_storage, capacity)`; you earn
`min(applied, demand[t])` toward supply (release beyond that day's demand
earns nothing extra) and pay `flood_cost_coef * spill**1.5` in flood cost.
Your objective is `supply - flood_cost_coef * sum(spill**1.5)`, **maximized**
and averaged (via the scoring below) over 10 fixed, seeded instances varying
skill-decay rate, pulse size/timing, release-rate cap, and demand-peak
placement. Several instances are larger / harder held-out cases.

## Scoring (deterministic)

The evaluator computes, itself, `q_base` (objective of the naive "release =
demand every day" policy) and `q_ideal = sum(demand)` (a generous,
essentially unreachable ceiling: perfect demand service *and* zero flood
damage at once — the release-rate cap makes some spill unavoidable on the
pulse instances). With `q_cand` your realized objective:

```
r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(eps, q_ideal - q_base), 0, 1 )
```

The reported **Ratio** is the mean of `r`; **Vector** lists per-instance
scores.

## Suggested strategies

1. **Match demand only** (baseline): no storage or flood management.
2. **Fixed high target**: keep storage near a constant high fraction of
   capacity, reacting only to today's forecast value.
3. **Skill-tracking buffer**: size standing headroom from how much surge
   volume could be hiding at each lead (`(1-skill[t]) * flash_risk *
   capacity`) plus the trusted visible signal, then scan ahead against the
   release-rate cap for the tightest safe storage level today.
4. **Stronger**: rolling re-optimization / a stochastic relaxation over
   plausible inflow realizations weighted by skill.
