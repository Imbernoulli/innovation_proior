# Market Garden: Rotation Plans as Soil Limit Cycles

A market garden has `P` plots, each with its own nutrient-state vector
`soil[k]`, `k = 0..K-1` (think N, P, K), clipped to `[0, cap[k]]`. You plan
what to plant on every plot for `T = 8` seasons. A **crop library** of `C`
crops is given; crop `c` has a family id, a per-nutrient requirement vector
`req[c]`, a per-nutrient depletion vector `depletion[c]`, a per-nutrient
replenishment vector `replenish[c]`, a `base_yield[c]` and a `price[c]`.

## Dynamics (identical for every plot, run independently per plot except for
## the market step below)

Each season, for a plot currently planting crop `c` on soil vector `s`:

```
yield_norm = clip( min_k ( s[k] / req[c][k] ), 0, 1 )      # 0 if req[k]==0, skip that k
pest_mult  = max(0, 1 - pest_coeff * pest_p)                # pest_p carried from last season
y_phys     = base_yield[c] * yield_norm * pest_mult
```

**Pest memory**: `pest_p` starts at 0 per plot. After the season's crop is
recorded, if this crop's family equals the *previous* season's family on
that plot, `pest_p = min(pest_cap, pest_p + pest_grow)`; otherwise
`pest_p = max(0, pest_p - pest_decay)`. Repeating a family builds pressure
that only decays when you switch families.

**Vector soil depletion/replenishment**: after harvest, for every `k`:
`s[k] = clip(s[k] - depletion[c][k]*yield_norm + replenish[c][k], 0, cap[k])`.
Depletion scales with how much was actually drawn; replenishment (e.g. a
nitrogen-fixing crop) is applied regardless of yield realized.

**Cross-plot diversification (market glut)**: within one season, let
`cnt[f]` be the number of plots planting family `f` that season (across the
whole garden, simultaneously). Revenue from a plot planting family `f` is
discounted by `glut_mult = min(1, glut_threshold / cnt[f])`. Planting the
same family on more plots than `glut_threshold` in one season floods the
local market and cuts the price on ALL of them, not just the excess ones.

**This season's revenue** for a plot = `price[c] * y_phys * glut_mult`.
Soil and pest state update using `yield_norm` — glut affects price only, not
the physical harvest.

## Input (stdin, JSON)

```json
{"P":.., "T":8, "K":.., "cap":[..], "glut_threshold":..,
 "pest_grow":.., "pest_decay":.., "pest_cap":.., "pest_coeff":..,
 "crops":[{"name":.., "family":.., "req":[..], "depletion":[..],
           "replenish":[..], "base_yield":.., "price":..}, ...],
 "init_soil":[[.. K values ..], ...]}   // one row per plot
```

## Output (stdout, JSON)

```json
{"plan": [[c_0_0, c_0_1, ..., c_0_7], [c_1_0, ...], ...]}
```
`plan[p][t]` is the crop index (`0..C-1`) planted on plot `p` in season `t`.
Wrong shape, an out-of-range/non-integer/non-finite index, a crash, a
timeout, or non-JSON output scores that instance `0.0`.

## Objective

Maximize total garden revenue: `sum` over all plots and all `T` seasons of
that plot-season's revenue, exactly as defined above.

## Scoring

The checker replays your `plan` with the exact dynamics above to get `F`,
and separately replays its own reference construction (monoculture of crop 0
everywhere, every season) to get `B` (always positive). It reports

```
Ratio = min(1, 0.1 * F / B)
```

averaged (equally) over 10 fixed, seeded instances that vary the plot count,
crop library and diversification threshold; two are larger held-out cases.
Matching the reference scores `0.1`; ten times the reference saturates at
`1.0`, but no reference solution reaches that ceiling — there is real
headroom above `strong`.

## Why the obvious plan underperforms

Planting whatever crop looks best *this season* on every plot pushes every
plot toward the same choice (they all start from identical soil), which
piles all plots onto one family at once (glut), repeats that family season
after season (pest spiral), and drains that crop's limiting nutrient with
nothing replenishing it — once a nutrient bottoms out with no replenishment
source in the rotation, it never recovers. The reward is not for the
best *single* decision but for finding a small rotation whose depletion and
replenishment cancel into a repeatable, sustainable orbit in soil-state
space, and then staggering that orbit's phase across plots so the garden as
a whole stays diversified every season.
