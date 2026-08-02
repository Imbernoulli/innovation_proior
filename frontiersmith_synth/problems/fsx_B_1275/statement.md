# Second Source, First Timed

## Problem

You run procurement for a component over a planning horizon of `T` periods. Each
period `t` needs exactly `D_t` units. There are `N` candidate suppliers, each
belonging to a **correlation group**: when a group is hit by a disruption event,
*every* supplier in that group delivers zero units for that period, no matter how
much you ordered. Exactly one supplier (index `0`) is already qualified to
deliver from period 1. Every other supplier `i` must first be **qualified**: if
you start qualifying it in period `s`, it becomes able to deliver from period
`s + lead_i` onward, at a one-time cost `qualcost_i` (paid regardless of whether
you end up using it).

Each supplier publishes **volume-discount tiers**: a list of `(threshold, price)`
pairs, sorted by threshold. An order of `q` units placed with supplier `i` in one
line item is priced at the *highest* threshold `<= q`, applied to all `q` units.
Placing one big order therefore claims a deeper discount than splitting the same
quantity across several smaller line items to the same supplier in the same
period — tier prices never increase with quantity, so consolidating a supplier's
per-period order is always at least as good as fragmenting it.

## Input (stdin)

```
T N G
V P
D_1 D_2 ... D_T
```
Then `N` lines, one per supplier `i = 0..N-1`:
```
group_i qualified_i lead_i qualcost_i ntiers_i  th_1 price_1  th_2 price_2  ...
```
`qualified_i` is 1 only for supplier 0. Tiers are sorted ascending by threshold;
`th_1` is always 0 (the undiscounted base price). Then:
```
E
period_1 group_1
...
period_E group_E
```
Each line means: every supplier in `group_e` delivers 0 units in `period_e`,
regardless of what was ordered.

## Output (stdout)

```
Q
supplier_i1 start_1
...
supplier_iQ start_Q
M
period_j1 supplier_j1 qty_j1
...
period_jM supplier_jM qty_jM
```
`Q` qualification actions (only for suppliers that start unqualified; each
supplier index at most once; `1 <= start <= T`), then `M` order line items
(`1 <= period <= T`, valid supplier index, `qty >= 0`).

## Feasibility

An order line to a supplier that is not yet qualified in that period, or whose
group is disrupted in that period, simply delivers 0 units at 0 cost (no charge
for a failed delivery) — it is not an error. What IS rejected (score 0): a
malformed token stream, `Q`/`M` out of range, re-qualifying an already-qualified
supplier, qualifying the same supplier twice, or any out-of-range index/period.

## Objective

For period `t`, let `deliv_t` be the total units actually delivered (summed
over lines that were qualified and undisrupted) and `cost_t` the sum of each
such line's own tier-priced cost. Then
```
value_t = min(D_t, deliv_t) * V  -  max(0, D_t - deliv_t) * P  -  cost_t
score    = sum_t value_t  -  sum of qualcost_i over qualification actions taken
```
Maximize `score`. `V` and `P` are given per instance.

## Worked example

`T=2, N=2, G=2, V=20, P=4`, demand `D = [10, 12]`. Supplier 0 (group 0,
qualified) has tiers `(0,10) (4,7) (8,5)`. Supplier 1 (group 1, lead 1,
qualcost 50) has tiers `(0,12) (4,9) (8,6)`. One disruption: period 2, group 0
(supplier 0 goes dark in period 2).

A solver that qualifies supplier 1 starting period 1 (ready from period 2) and
orders 10 units from supplier 0 in period 1 and 12 units from supplier 1 in
period 2: period 1 clears supplier 0's top tier (price 5): `10*20 - 10*5 = 150`.
Period 2 clears supplier 1's top tier (price 6): `12*20 - 12*6 = 168`. Total
`150 + 168 - 50 (qualcost) = 268`. Single-sourcing supplier 0 for both periods
instead scores `10*20-10*5 + (0 - 12*4) = 150 - 48 = 102` — disruption wipes out
period 2 entirely because no one else was ready in time. *(Illustrative FORM
only — the shape of a strong strategy on the real test data differs.)*

## Constraints

`1 <= T <= 25`, `2 <= N <= 10`, `1 <= G <= N`, `1 <= D_t <= 200`,
`0 <= lead_i <= T`, `0 <= qualcost_i, price <= 10^5`, `0 <= E <= T*G`. Time
limit 5s, memory 512MB.
