# Trend-Tail Slotting: p90 Order-Wave Picking Distance

## Problem
A warehouse stocks `N` SKUs on a rack of `A` aisles, each with `L` slots at
depths `1..L` (so `N = A*L`). Slot index `t` in `[0, N-1]` corresponds to
aisle `t // L + 1` and depth `t % L + 1`. The depot sits at horizontal
position `0`, aisle `k` sits at horizontal position `k` (spacing `W`).

You must place every SKU into exactly one slot (a bijection SKU -> slot).
Shopping fashions shift: SKUs are pulled together into orders by seasonal
co-purchase patterns that differ from one order wave (scenario) to the next.
The warehouse ships `K` such scenarios. For each order wave `s`, a FIXED
picker policy processes every order (a small list of required SKUs) as a
separate round trip from the depot:

- Let the order's required SKUs land in slots that touch aisle set `U`.
- Horizontal cost: `2 * W * max(U)` (the picker sweeps out from the depot to
  the farthest required aisle and back; this automatically passes every
  aisle in between).
- Vertical cost: for every aisle `a` in `U`, the picker walks in to the
  DEEPEST required slot in that aisle and back out: contributes
  `2 * depth_max(a)`.
- The order's distance is the sum of these two costs. A scenario's total
  cost is the sum of its orders' distances.

The picker policy itself is fixed and never changes -- your only lever is
WHERE each SKU sits. Your objective is the **90th percentile (p90) of the
`K` scenario totals**, using the nearest-rank definition: sort the totals
ascending and take the one at index `ceil(0.9*K) - 1` (0-indexed). Minimize
this value. Because it is a percentile, not a mean, only the handful of
worst-case scenarios matter -- a layout that is excellent on typical
scenarios but bad on the tail scores poorly, and vice versa.

## Input (stdin)
```
N A L K W
```
Then `K` blocks, one per scenario, each:
```
q
s_1 sku_1 sku_2 ... sku_{s_1}
...
s_q sku_1 ... sku_{s_q}
```
`q` = number of orders in that scenario; each order line starts with its
size `s_i` (2..6) followed by that many distinct SKU ids in `[0, N-1]`.

## Output (stdout)
```
N
p_0
p_1
...
p_{N-1}
```
`p_i` is the slot assigned to SKU `i`. The `N` values must be a permutation
of `0..N-1`.

## Feasibility
- The declared count on line 1 must equal `N`.
- Exactly `N` further integer tokens, each finite and in `[0, N-1]`.
- The `N` values must be pairwise distinct (a permutation).

Any violation scores `Ratio: 0.0`.

## Scoring
The checker builds an internal baseline `B` = the p90 scenario cost under the
IDENTITY placement (SKU `i` in slot `i`). Let `F` be the p90 scenario cost of
your placement. Since this is a minimization objective:
```
sc    = min(1000, 100 * B / F)
Ratio = sc / 1000
```
Matching the identity baseline scores `0.1`; a 10x-lower p90 cost caps at
`1.0`. Note the identity baseline does NOT know anything about SKU velocity
or co-purchase structure, so it is easy to beat -- reaching a good score
requires understanding both which scenarios end up in the tail and which
SKUs must sit together to keep those specific scenarios cheap.

## Constraints
- `16 <= N <= 600`, `2 <= A,L`, `6 <= K <= 20` across the test ladder.
- Deterministic exact scoring; no randomness or timing in the score.

## Example
`N=4, A=2, L=2, K=1, W=2`. One scenario with one order `{0,1}`. Identity
placement: SKU0->slot0 (aisle1,depth1), SKU1->slot1 (aisle1,depth2). Aisles
used = `{1}`, depth_max=2. Distance = `2*2*1 + 2*2 = 8`, so `B=8`. A
placement putting SKU0->slot0, SKU1->slot2 (aisle2,depth1) instead gives
aisles `{1,2}`, so distance = `2*2*2 + 2*(1+1) = 12` (worse: spreading the
pair across aisles costs more even though depth improved) -> `F=12`,
`Ratio = min(1000, 100*8/12)/1000 = 0.0667`.
