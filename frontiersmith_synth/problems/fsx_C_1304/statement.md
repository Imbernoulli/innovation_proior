# Watering a Field That Remembers Last Week

A field is irrigated over a `T`-day growing season. Water — rain or pumped
irrigation — does not reach the crop's roots instantly. It first lands in a
thin **surface layer** (capacity `Cs`); only a fraction `alpha` of whatever sits
in the surface layer **percolates down** into the deep **root-zone reservoir**
(capacity `Cr`) each day. The rest stays in transit for tomorrow, or is lost as
runoff the moment the surface layer is already full. The crop drinks only from
the root zone. This two-layer bucket is the field's **storage memory**: water
applied today is not fully usable today, and water applied when the surface is
already saturated (e.g. right before a rainstorm) is wasted.

The crop passes through four **phenological stages** (`0`=establishment,
`1`=vegetative, `2`=flowering, `3`=maturation), each with its own minimum
comfortable root-zone moisture fraction and its own **sensitivity weight** —
flowering is by far the most demanding and the most sensitive to shortfall.
Pumping costs money at a **time-varying tariff** that has nothing to do with
when the crop is thirsty.

Irrigating to one fixed moisture target every day is the standard textbook
approach — and it is exactly what wastes water right before a forecasted
downpour, and exactly what under-waters the crop during the critical
flowering window when the target doesn't account for the transfer lag or the
price spike.

## Your job

Submit a full-season **irrigation schedule**: how many mm to pump on each of
the `T` days, chosen with full knowledge of the season's rain forecast,
evapotranspiration demand, tariff schedule, and stage layout (all given to
you).

### Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**,
write ONE JSON object (your answer) to **stdout**. Runs in an isolated
subprocess.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a season-long schedule ...
print(json.dumps({"irrig": irrig}))
```

### Public instance (stdin)

```json
{
  "name": "field101", "T": 40,
  "rain":   [r_0, ..., r_{T-1}],     // mm forecast to fall, per day
  "et":     [e_0, ..., e_{T-1}],     // mm potential evapotranspiration demand, per day
  "tariff": [c_0, ..., c_{T-1}],     // $ per mm pumped, per day
  "stage":  [s_0, ..., s_{T-1}],     // phenological stage id 0..3, per day
  "params": {
    "Cs": 45.0, "Cr": 140.0, "alpha": 0.35,      // surface cap, root cap, daily percolation fraction
    "theta_max": 0.93,                            // waterlogging threshold (fraction of Cr)
    "stage_theta_min": [t0, t1, t2, t3],          // minimum comfortable root moisture per stage
    "stage_sensitivity": [w0, w1, w2, w3],        // yield-loss weight per stage
    "S0": 5.0, "R0": 70.0,                        // starting surface / root moisture (mm)
    "max_irrig_per_day": 35.0,                    // pump cap per day (mm)
    "cost_scale": 0.16, "Y_max": 100.0            // $-to-yield conversion, max yield units
  }
}
```

### Answer (stdout)

```json
{ "irrig": [x_0, ..., x_{T-1}] }   // length T, each 0 <= x_t <= max_irrig_per_day
```

Any wrong length, non-numeric or out-of-range entry, non-finite value, crash,
timeout, or non-JSON output scores that instance `0.0`.

## Objective

**Maximize** `yield_value - cost_scale * total_pumping_cost`, averaged (after
normalization) over a fixed, seeded family of 10 instances of varying length,
percolation rate, weather pattern, and tariff schedule; several are deliberately
adversarial (a big rain event right after a dry spell, a slow percolation rate
that demands a long lead time before flowering, a tariff spike that coincides
with flowering) and some are larger held-out cases.

## Scoring (deterministic)

The evaluator re-simulates the TRUE two-layer water balance for your schedule
day by day (surface fill → percolation into root zone with the `alpha`
throughput cap → root-zone draw-down by `et`), accumulating a
stage-sensitivity-weighted deficit penalty (plus a smaller waterlogging
penalty above `theta_max`) into `yield_value`, and the pumping cost from
`tariff * irrig`. Your raw objective is then normalized against two references
the evaluator computes itself: a weak, single-bucket fixed-target controller
(anchored to `0.1`) and a loose, generally-unreachable ideal (anchored to
`1.0`, priced at an artificial discount below the cheapest tariff day — no
schedule can truly reach it). Beating the weak controller scores above `0.1`;
doing worse scores below it.
