# Shared Stockpot Chain: A Forest of Brew Orders

## Problem
A potion kitchen keeps `m` base ingredient-potions in one long shelf, indexed
`0..m-1`. Ingredient `i` has a *strength profile* pair `(d[i], d[i+1])`: it can be
**combined** with an adjacent partial brew the way matrix multiplication chains
(the combine of a run covering `[a,k)` with an adjacent run covering `[k,b)` is only
valid when their boundary strengths line up at `k`, and it costs exactly
`d[a] * d[k] * d[b]` units of kettle-work). Combining runs is associative: the
*potion* you get from brewing ingredients `L..R-1` in any legal nesting is always the
same potion, only the kettle-work to get there differs.

The kitchen has `Q` standing **brew orders**. Order `q` asks for the single potion
obtained from ingredients `[L_q, R_q)`. Orders routinely overlap: many orders need
overlapping (not necessarily identical) stretches of the shelf. Your job is to plan
brewing for **all `Q` orders together** so that any partial brew (any sub-run) that
several orders need is cooked **once** and reused, not recooked per order.

## Input (stdin)
```
m
d[0] d[1] ... d[m]
Q
L_1 R_1
L_2 R_2
...
L_Q R_Q
```
`1 <= m <= 80`, `20 <= d[i] <= 200`, `1 <= Q <= 50`, `0 <= L_q < R_q <= m`,
`R_q - L_q >= 2`.

## Output (stdout)
Declare a DAG of combine-nodes, then name each order's result node. First `N`, the
node count, then `N` node records (each is either `L i` for a leaf = ingredient `i`,
or `S c1 c2` for a combine of two **earlier** node ids `c1 < id`, `c2 < id`), then `Q`
integers `r_1 ... r_Q`: the node id that is order `q`'s result. Whitespace (including
newlines) between tokens is free.

A node's covered range `[lo,hi)` is *derived*, not asserted: leaf `L i` covers
`[i,i+1)`; `S c1 c2` requires `hi(c1) == lo(c2)` (the two children must be exactly
adjacent) and then covers `[lo(c1), hi(c2))`.

## Feasibility
- `1 <= N <= 400000`. You need not declare a leaf for an ingredient that no order's
  range ever touches — only nodes reachable from some `r_q` are ever scored.
- Every child id referenced by an `S` node must be a *strictly earlier* node id.
- Every `S` node's two children must be exactly adjacent (no gap, no overlap); by
  induction this forces every node's range to tile a contiguous run of real leaves,
  so any node claiming range `[L_q,R_q)` genuinely combines exactly ingredients
  `L_q..R_q-1` in some legal nesting — there is no way to under-declare and still
  match a query's required range.
- For every order `q`, `r_q` must be a valid node id whose derived range is exactly
  `[L_q, R_q)`.

Any violation, parse error, out-of-range token, or non-finite/non-integer token scores
`Ratio: 0.0`.

## Objective
Minimize total kettle-work. Walk the union of all `Q` result nodes' dependencies
(every node reachable from any `r_q`); **each distinct node id in that union is
charged exactly once**, however many orders reach it. A leaf costs `0`; an `S c1 c2`
node costs `d[lo(c1)] * d[hi(c1)] * d[hi(c2)]`. `F` = the sum over the reachable
union. Declaring the same sub-range via *different* node ids counts them as different
(unshared) computations — reuse only pays off when orders are routed through the
*same* node id.

## Scoring
The checker independently builds `B` = the cost of solving every order completely
independently via the dumbest possible left-to-right fold (no sharing at all). With
your total `F`:
```
Ratio = min(1, 0.1 * (B / F) ** 0.85)
```
Ratio `0.1` reproduces the naive per-order baseline. Growing the shared-reuse economy
(lower `F` for the same `B`) raises the score smoothly; the exponent keeps headroom
open even when many orders collapse onto a few shared kettles.

## Constraints
Time limit 5s, memory 512MB. Feasibility checks and the totals `F` and `B` are all
exact integer arithmetic; the final ratio (`(B/F)**0.85`, printed to 6 decimals) is
one deterministic floating-point formula applied to those exact integers afterward —
no randomness, no wall-clock, no source of run-to-run variation anywhere in scoring.

## Example
`m=3`, `d=[10,20,5,15]`, one order `[0,3)`. Output:
```
5
L 0
L 1
L 2
S 0 1
S 3 2
4
```
Node `3` (`S 0 1`) covers `[0,2)`, cost `d[0]*d[1]*d[2] = 10*20*5 = 1000`. Node `4`
(`S 3 2`) covers `[0,3)`, cost `d[0]*d[2]*d[3] = 10*5*15 = 750`. Both are reachable
from the one root (`4`), so `F = 1000 + 750 = 1750`. The checker's independent
left-to-right fold for this single order builds the identical plan, so `B = 1750`
and `Ratio = min(1, 0.1*(1750/1750)**0.85) = 0.1`. With `Q` orders sharing structure,
a plan that lands `F` at a quarter of `B` scores `0.1*4**0.85 ≈ 0.31`; a plan that
lands `F` at a tenth of `B` scores `0.1*10**0.85 ≈ 0.71` — pushing `F` lower by
sharing more of the reachable union raises the score, whether that reduction comes
from a single order's own optimal split or from many orders reusing the same node.
