# Raw-Material Order Book: Hitting a Product Mix Through Batch Yields

## Story
A small plant makes **3 products** (indices `0,1,2`) from **`P` raw-material types**.
Raw type `j` is processed in whole **batches** of `lot[j]` raw units: every *complete*
batch of `j` yields the fixed vector `yield[j]` (3 nonnegative integers, summing to
more than 0). Raw units bought beyond the last complete batch are **wasted** — no
output, but still paid for. Buying and processing all cost `cost[j]` per raw unit
(charged on every unit you buy, wasted or not).

Every raw type also has a **full-yield threshold** `thresh[j]`: only its first
`thresh[j]` batches run at the full `yield[j]` rate. Every batch beyond that runs
**degraded**, producing `floor(yield[j][k] * degrade_num[j] / degrade_den[j])` of each
product `k` instead (`0 < degrade_num[j] < degrade_den[j]`).

You choose an integer purchase quantity `order[j]` for each raw type
(`0 <= order[j] <= maxOrder[j]`). Your goal: get the plant's **output ratio** as close
as possible to a given **target ratio**, while not overspending.

## Task
Write a **standalone program**: read ONE JSON instance from `stdin`, write ONE JSON
answer to `stdout`.

### Public instance (stdin)
```json
{ "P": 5, "K": 3,
  "lot":  [...P positive ints...],
  "yield": [[y0,y1,y2], ...P rows, nonneg ints, each row sums > 0...],
  "cost": [...P positive ints, per raw UNIT bought...],
  "thresh": [...P positive ints...],
  "degrade_num": [...P ints...], "degrade_den": [...P ints, num < den...],
  "maxOrder": [...P positive ints, purchase cap per type...],
  "target": [t0, t1, t2] }
```

### Answer (stdout)
```json
{ "order": [order_0, ..., order_{P-1}] }
```
`P` integers, `0 <= order[j] <= maxOrder[j]`. Any violation — wrong length, wrong
type, out of range, NaN/inf, a boolean, a crash, a timeout, or non-JSON output —
scores **0.0** on that instance.

## Transition (deterministic)
For raw type `j`: `nb = order[j] // lot[j]` complete batches run; you pay
`order[j] * cost[j]` regardless. Batches `1..min(nb, thresh[j])` each add `yield[j]`
to the running output vector; any further batches (`nb - thresh[j]` of them, if
positive) each add `floor(yield[j][k]*degrade_num[j]/degrade_den[j])` per product `k`.

## Objective & scoring (deterministic)
Let `output` be the resulting 3-vector and `S = sum(output)`. If `S > 0`,
`ratio[k] = output[k]/S` and `dev = sum_k |ratio[k] - target[k]|` (in `[0,2]`); if
`S == 0` (you produced nothing), `dev = 2.0` — ordering nothing can never match a
target. Let `spend = sum_j order[j]*cost[j]` and `spend_cap = sum_j maxOrder[j]*cost[j]`.
Your per-instance objective (**lower is better**) is:
```
obj = dev + GAMMA * (spend / spend_cap),   GAMMA = 0.5
```
The evaluator separately computes a weak reference `b` (buy exactly one batch of the
single cheapest-per-unit raw type, ignoring the target) and maps your objective to a
per-instance score `r = min(1, 0.1 * b / max(obj, 1e-9))`. The final score is the mean
of `r` over **10** fixed seeded instances.

## Why it is open-ended
Buying more of a single raw type run entirely within its full-yield threshold leaves
*its own* output ratio unchanged — pushing batches past the threshold nudges it only
slightly (whole-product-unit flooring on the degraded share). So the set of ratios
reachable by any non-negative purchase combination is well approximated by the convex
hull, on the product simplex, of the raw types' own normalized full-yield directions;
combining two raw types traces close to the segment between their two directions. Some
target ratios lie far enough outside that hull that no purchase, however combined, can
ever get close; the right move is to aim for the nearest reachable point instead of
chasing the unreachable one at ballooning cost — but the final choice of batch counts
must still be checked against the real simulated output, since the hull is only a
first-order guide once degradation is in play. Picking, for each product independently, whichever
single raw type looks like the best per-cost source and buying it in proportion to
that product's target share ignores that most raw types are mixed recipes (summing
independent per-product picks double-counts side yields) and ignores that a source
purchased past its threshold quietly yields less than its sticker rate — both drag the
realized ratio and the spend away from what was intended. Trading off which raw types
to combine, how far past a threshold to push a cheap source versus topping up with a
pricier steady one, and how much total scale to buy (larger purchases shrink the
*relative* rounding error from batch integrality but cost more) has no easy optimum.

## Isolation
Your program runs in a fresh sandboxed subprocess and only ever sees the public
instance above.
