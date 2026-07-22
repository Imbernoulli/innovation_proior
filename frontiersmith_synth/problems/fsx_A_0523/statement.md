# Sweet-Spot Boiler Fleet: Winter Heat at Least Fuel

A district heating plant must meet a winter heat demand over `T` time steps with a fleet of
`K` boilers. Each step you choose which boilers are **online** and how hard each runs.
Boilers are fuel-hungry to keep lit and inefficient off their sweet spot, so *which* boilers
you commit and *how* you load them must be decided together.

## Fuel model
Boiler `i` has capacity `C_i`, minimum stable output `pmin_i`, no-load fuel `c_i`,
specific-fuel scale `a_i`, curvature `b_i`, and sweet-spot load fraction `x_i` (near 0.7).
If online at output `o` (with `pmin_i <= o <= C_i`) it burns

```
fuel_i(o) = c_i + a_i * o * (1 + b_i * (o/C_i - x_i)^2)
```

Efficiency (heat per fuel) peaks at load fraction `x_i` and falls off on both sides; the
constant `c_i` is paid every step the boiler is online. A boiler that is **offline** produces
`0` and burns `0`.

## Input (stdin)
```
line 1:  T K
line 2:  D[0] D[1] ... D[T-1]          integer heat demand per step
next K lines:  C_i pmin_i c_i a_i b_i x_i ramp_i minup_i mindown_i
```

## Output (stdout)
`T` lines, each with `K` numbers: line `t` gives `o[t][0] ... o[t][K-1]`, the output of every
boiler at step `t`. A boiler is **online** at step `t` iff `o[t][i] > 0`.

## Feasibility (all must hold; any violation scores 0)
1. For every `t,i`: either `o[t][i] = 0` (offline) or `pmin_i <= o[t][i] <= C_i` (online).
2. **Demand met:** `sum_i o[t][i] >= D[t]` for every `t` (over-production is allowed but
   wastes fuel).
3. **Ramp:** if boiler `i` is online at both `t-1` and `t`, then `|o[t][i]-o[t-1][i]| <= ramp_i`.
   Ignition (offline→online) and shutdown (online→offline) are exempt.
4. **Min up / min down:** every maximal online stretch of boiler `i` has length `>= minup_i`,
   and every maximal offline stretch has length `>= mindown_i` — except a stretch that touches
   the start or the end of the horizon. All boilers start **offline** before step 0.

## Objective (minimize)
Total fuel `F = sum over t, i of fuel_i(o[t][i])`.

## Scoring
Let `B` be the fuel of a reference schedule the checker builds itself: **every boiler online
for the whole horizon**, each carrying a share of demand proportional to its capacity
(raised to `pmin_i` when that share is smaller). Your score is

```
ratio = min(1000, 100 * B / F) / 1000
```

so reproducing the reference scores `0.1` and burning `10x` less caps at `1.0`. Higher is
better; there is deliberate head-room above any simple construction.

## Why it is hard
No-load fuel makes keeping a boiler lit expensive, while the sweet-spot penalty makes running
a saturated boiler expensive too. Merit-order thinking — commit a fixed fleet sized for the
peak and split demand across it — pays full no-load through the long shoulders and loads
boilers off their sweet spot. The cheapest marginal unit of heat often comes from **de-loading
a boiler back toward its sweet spot and lighting (or shedding) another**, phased along the
demand ramp so ramp and min-up/min-down limits are respected. Commitment and loading must be
co-optimized, not layered.

## Example
`T=2, K=2`, demand `[40, 60]`. Boiler 0: `C=50, pmin=15, c=8, a=1.0, b=3.0, x=0.7`.
Boiler 1: `C=50, pmin=15, c=8, a=1.0, b=3.0, x=0.7`. Meeting step 0 with a single boiler at
`o=40` (`x=0.8`) costs `8 + 40*(1+3*0.01)=49.2`; splitting `20/20` costs `2*(8+20*(1+3*0.09))
=66.9` — one boiler near its sweet spot beats two half-loaded ones once the second no-load is
counted. At step 1, `60` needs both boilers; `30/30` (`x=0.6`) costs `2*(8+30*1.03)=77.8`.
Constraints (ramp, min-up/down) decide whether the second boiler may be lit only for step 1.

Constraints: `2 <= K <= 12`, horizons up to a few thousand steps, `time 2–5 s`, memory `512 MB`.
