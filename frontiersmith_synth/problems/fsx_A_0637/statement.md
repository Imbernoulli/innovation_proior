# Tideline Commons: Multi-Zone Fishery Quota Setting

A fishing cooperative manages several offshore **zones**. Each zone `z` holds a
fish stock `S` that regenerates every step by a logistic-style law: growth is
**largest near `S = K/2`** (`K` = carrying capacity) and shrinks toward zero as
`S` nears 0 or `K`. Every step the cooperative sets a **harvest quota** per
zone; the quota is removed first, and only *then* does the remainder
regenerate. Over `T` steps, **maximize total fish landed** across all zones.

## The nonlinear catch

A policy that mines a zone down to a low stock harvests far less regrowth per
step than one that holds the stock near `K/2` and skims only the surplus.

## The invisible trap

Every zone also has a **collapse threshold**: an invisible floor, never
revealed to you. If, after harvesting, a zone's stock ever drops **below its
own threshold**, that zone **collapses for the rest of the episode** — its
regeneration rate drops to a residual `collapse_growth_mult` (2%) of normal,
forever, destroying nearly all future yield from that zone. You are never told
the exact threshold, but you ARE guaranteed, for every zone in every instance,
that **no threshold exceeds `max_threshold_frac` (45%) of that zone's own
`K`**.

## Candidate program contract

Your solution is a **standalone program**: read ONE JSON object (the public
instance) from stdin, write ONE JSON object (your answer) to stdout. It runs
in an isolated subprocess and sees only the public instance below.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a harvest plan ...
print(json.dumps({"harvest": harvest}))
```

### Public instance (stdin)

```json
{
  "name": "zone101",
  "T": 40,
  "n_zones": 4,
  "collapse_growth_mult": 0.02,
  "max_threshold_frac": 0.45,
  "zones": [
    {"K": 210.4, "r": 0.31, "S0": 88.7},
    ...
  ]
}
```

`K` = carrying capacity, `r` = growth rate, `S0` = starting stock, per zone.

### Answer (stdout)

```json
{ "harvest": [[h_00, h_01, ..., h_0(Z-1)],
              [h_10, h_11, ..., h_1(Z-1)],
              ...
              [h_(T-1)0, ..., h_(T-1)(Z-1)]] }
```

`harvest` must be a list of **exactly `T`** rows, each a list of **exactly
`n_zones`** non-negative finite numbers: `h_tz` is what you request from zone
`z` at step `t`; a request exceeding available stock is silently **clipped**.
Any shape/type violation (wrong length, negative, NaN/Infinity, non-JSON), a
crash, or a timeout makes that instance score `0.0`.

## Dynamics (per zone, per step; evaluator-side, uses the TRUE hidden
threshold — never sent to you)

```
applied   = min(h_tz, S)                         # clip to available stock
S_after   = S - applied
collapsed = collapsed OR (S_after < theta_z)      # sticky: once true, stays true
mult      = collapse_growth_mult if collapsed else 1.0
growth    = mult * r_z * S_after * (1 - S_after / K_z)
S         = clip(S_after + growth, 0, K_z)
```

Your total catch is the sum of `applied` over every zone and every step.

## Objective

**Maximize** total catch across a fixed, seeded family of 10 instances (3–6
zones, 30–60 steps, varying `K`, `r`, `S0`, and hidden thresholds — some
instances plant zones whose true threshold sits well above what a
one-size-fits-all "safe floor" guess would assume).

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `q_base`  = 0 (a "harvest nothing" reference),
- `q_ideal` = `sum_z [ T * r_z*K_z/4 + 0.3*K_z ]` — a loose, **unreachable**
  ideal (every zone at its max-growth point for the whole horizon *plus* a
  fictional bonus windfall — no real trajectory achieves both),
- `q_cand`  = your realized total catch,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, q_ideal - q_base), 0, 1 )
```

The reported **Ratio** is the mean of `r` over all instances; the **Vector**
lists the per-instance scores.

## Suggested strategies

1. **Fixed-rate skim** (weak): cull a constant fraction of stock every step.
2. **Static generic-floor greedy**: one safety buffer for every zone; wastes
   surplus, and can walk through a threshold higher than the buffer.
3. **Max-growth-point holding**: harvest only the surplus above `K/2`,
   provably above every zone's guaranteed threshold bound.
4. **Refine further**: tune warm-up, or exploit the final step's lack of a
   future to protect, for extra yield above strategy 3.
