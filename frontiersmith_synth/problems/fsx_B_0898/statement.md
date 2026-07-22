# Shopkeeper Survives Every Embargo in the Almanac

You run a shop over a horizon of `H` days. The almanac lists every day's
demand for the whole horizon — demand is exogenous, never reacting to what
you order. What it cannot tell you is which days a trade **embargo** will
close the border: on an embargoed day, **zero supply** gets through, so any
shipment due that day is lost outright (no goods, no charge). Before an
embargo, caravan gossip carries a numeric **precursor signal** that ramps up
in the days just before it starts and fades back to background chatter
partway through — an imperfect but genuine warning.

You submit **one ordering policy**, shared across every timeline in the
instance. Each day, for each timeline, the (evaluator-run) shop reviews its
inventory position (on-hand plus outstanding orders) and orders up to a
target:

```
target(t) = hoard_target   if precursor_signal[t] >= trigger, OR a trigger
                             fired within the last cooldown_days days
          = base_target    otherwise
```

Orders placed on day `t` arrive on day `t + lead_time` unless that day falls
inside an embargo window, in which case the shipment is lost. Demand not met
from on-hand stock is a lost sale (`stockout_penalty` per unit, on top of
forgone revenue). Holding cost is charged on end-of-day on-hand stock.

**The instance's score is the minimum profit ratio across its timelines.** A
policy is only as good as its worst timeline: permanent hoarding starves the
calm timelines with holding cost exactly as badly as never hoarding
bankrupts the embargo ones. The only way to win the minimum is to fund
embargo insurance out of calm-timeline efficiency — hoard only when the
sweep's own precursor signal actually says to.

## Candidate program contract

Standalone program: read ONE JSON public instance from **stdin**, write ONE
JSON answer to **stdout**. Runs in an isolated subprocess.

### Public instance (stdin)

```json
{
  "name": "embargo03", "n_timelines": 5, "horizon": 36, "lead_time": 3,
  "price": 6.0, "unit_cost": 3.0, "holding_rate": 0.12, "stockout_penalty": 4.0,
  "init_stock": 24.0,
  "timelines": [
    {"demand": [d_0, ..., d_35], "precursor_signal": [p_0, ..., p_35]},
    ...
  ]
}
```
`demand[t] >= 0` is that day's realized demand (fully revealed, every day).
`precursor_signal[t] >= 0` is that day's early-warning reading (also fully
revealed). No timeline's embargo windows are revealed — only the signal
hints at them.

### Answer (stdout)

```json
{"base_target": 30.5, "trigger": 1.8, "hoard_target": 62.0, "cooldown_days": 6}
```
`base_target`, `hoard_target`: finite numbers in `[0, 1e6]`. `trigger`: any
finite number. `cooldown_days`: a non-negative integer `<= 60`. Any missing
key, wrong type, out-of-range value, a crash, a timeout, or non-JSON output
scores that instance `0.0`.

## Scoring (deterministic)

For each instance the evaluator computes two references itself, using the
**same** order-up-to simulator as your policy:

- `BASE` — min-over-timelines profit of the "obvious" recipe: a constant
  order-up-to level from the sample mean/std of demand, which **never**
  hoards — blind to the signal and to embargoes.
- `UB` — min-over-timelines profit of an oracle handed the **true** (hidden)
  embargo windows, which pre-builds an exactly-sized buffer starting
  `lead_time` days before each embargo and holds it through its end. It uses
  information no causal policy can access, so `UB` is a valid ceiling no real
  submission can be expected to reach.

```
r = clamp( 0.1 + 0.9 * (MIN_yours - BASE) / denom, 0, 1 )   # denom = max(UB-BASE, small stability floor)
```

Matching `BASE` scores ≈ 0.1; approaching `UB` scores near 1.0; worse than
`BASE` scores below 0.1. **Ratio** is the mean of `r` over 10 fixed
instances; **Vector** lists per-instance scores. Several instances mix calm
and embargo timelines (early, long, doubled, or with a demand-only spike
carrying **no** precursor warning) — exactly where ignoring the signal costs
you in the minimum.

## Suggested strategies

1. Order up to a bare minimum (mean demand times lead time), never hoard —
   the do-nothing baseline.
2. Textbook base-stock from the sample mean/std of demand, still never
   hoarding — the obvious first move.
3. Also read the precursor signal's own distribution (across the whole
   instance) to set a statistically-separated trigger, size a hoard buffer
   from the demand data, and keep it active a few days past each trigger
   since the signal fades before the embargo itself ends.
