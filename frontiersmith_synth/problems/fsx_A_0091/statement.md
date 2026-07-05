# Sunrise Bakery Standing-Order Grid: Spoilage-Free Activation

## Problem
A regional bakery cooperative plans its **standing orders** on an `m x m` grid. A slot is
addressed by a pair `(w, t)` where `w` in `0..m-1` is a **warehouse index** and
`t` in `0..m-1` is a **delivery-window index**. Activating a slot means committing a
recurring shipment from warehouse `w` in window `t`.

Some slots are **embargoed** (a labor or logistics hold this season) and may not be used.
You may **activate** any subset of the remaining slots, but the shared cold-chain has a
failure mode called a **spoilage cascade**: it triggers whenever three *distinct* active
standing orders form a right-angle "corner"

```
(w, t) , (w + d, t) , (w, t + d)      for some integer d != 0,
```

i.e. one order at the corner, a second in the **same window** `d` warehouses away, and a
third in the **same warehouse** `d` windows away — the two legs having equal length `d`.
Here `d` may be positive or negative, and all three coordinates must lie in `0..m-1`.

The cooperative wants as many standing orders running as possible without ever forming a
spoilage cascade. This is exactly the **corner-free set** condition on the grid
`{0,...,m-1}^2` (no axis-aligned isosceles right-angle corner).

## Input (stdin)
```
m
e
<e lines, each "w t" = an embargoed slot>
```
`m` is the grid side; `e` is the number of embargoed slots. Embargoed pairs are distinct
and every coordinate lies in `0..m-1`.

## Output (stdout)
```
k
<k lines, each "w t" = an activated standing order>
```
Print the number of activated orders `k`, then the `k` addresses, one per line.

## Feasibility
An output is valid iff **all** hold:
- each line is two integers `w t` with `0 <= w < m` and `0 <= t < m`;
- the `k` addresses are pairwise distinct;
- no activated address is an embargoed slot;
- no three distinct activated orders form a spoilage cascade (corner).
Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = k`, the number of activated standing orders (a corner-free subset of the
`m x m` grid avoiding embargoed slots).

## Scoring
Let `B` be the size of the checker's own trivial construction: **delivery window `t = 0`**,
i.e. the slots `{ (w, 0) : 0 <= w < m }` restricted to non-embargoed slots. A single window
shares one `t`, so it can never close a corner (a corner needs a leg into a *different*
window); hence this set is always valid. The generator never embargoes window `0`, so
`B = m`. With maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing window `0` scores `Ratio = 0.1`; a corner-free set `10x` larger caps at `1.0`.

## Constraints
- `6 <= m <= 15`, so the grid has `36 <= m^2 <= 225` slots.
- Delivery window `t = 0` is never embargoed (the baseline is always available).
- Time limit 5s, memory 256m.

## Example
Suppose `m = 6` and nothing is embargoed. Window `0` is `{(0,0),(1,0),...,(5,0)}` with
`B = 6` and is spoilage-free, scoring `0.1`. A corner-free set of size `12` gives `F = 12`,
`sc = 100*12/6 = 200`, `Ratio = 0.200`.
