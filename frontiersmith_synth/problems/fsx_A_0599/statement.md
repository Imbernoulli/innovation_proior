# Matched-Batch Ladder: Dividers That Hold at Every Tolerance Corner

## Problem
You are laying out `K` resistive voltage dividers on one board from a catalog of `M`
distinct resistor values. Divider `k` must produce output ratio close to a target
`r_k` in `(0,1)`. A divider is a **top** resistance and a **bottom** resistance; its
output ratio is `Rbot / (Rtop + Rbot)`.

Each resistance is a series pile of catalog parts: you choose, for the top and for the
bottom of every divider, how many copies of each catalog value to use (nonnegative
integers). A resistance's value is the sum of `count * value` over the catalog.

All parts come from the same production batch. The board must meet spec at **every
tolerance corner**. A corner is a sign vector `s` over the `M` catalog values: at that
corner every part of value `v_j` deviates *together* to `v_j * (1 + s_j * t)`
(same-valued parts are perfectly correlated — they came off the same reel). The input
lists `C` corner vectors and the tolerance `t`.

## Input (stdin)
```
K M P C TPM
r_1 ... r_K            (target ratios, decimals in (0,1))
v_1 ... v_M            (distinct positive catalog values)
s_1[1..M]             (corner 1: each entry -1 or +1)
...
s_C[1..M]             (corner C)
```
`t = TPM / 1000`. `P` is the total part budget.

## Output (stdout)
`K` lines. Line `k` has `2M` nonnegative integers: the top counts
`a_{k,1..M}` then the bottom counts `b_{k,1..M}` for divider `k`.

## Feasibility
- Exactly `K` lines of exactly `2M` integers, each in `[0, P]`.
- Every divider has a nonempty top (`sum a_k >= 1`) and nonempty bottom (`sum b_k >= 1`).
- Total parts used `sum over all a,b <= P`.
Any violation scores `0`.

## Objective (minimize)
For divider `k` and any sign vector `s`, let
`Rtop(s) = sum_j a_{k,j} v_j (1 + s_j t)`, `Rbot(s)` likewise, and
`ratio_k(s) = Rbot(s) / (Rtop(s) + Rbot(s))`.
Your board is scored by its **worst-corner spec deviation**
```
F = max over k, over s in { nominal (all zero) } U { the C listed corners }  of  | ratio_k(s) - r_k |
```
Smaller `F` is better.

## Scoring
The checker builds an internal baseline `B` = the worst deviation of the all-halves
board (every tap = two matched units of `v_1`, exactly `1/2`, corner-invariant). Then
```
Ratio = min(1000, 100 * B / F) / 1000
```
So reproducing the baseline scores ~`0.1`; a `10x`-smaller worst deviation caps at `1.0`.

## Constraints
`4 <= K <= 8`, `6 <= M <= 12`, `64 <= C <= 128`, `TPM = 50` (5%). `P = 32K`. Time 5s.

## Example (worked score)
Say `K=1`, target `r_1 = 0.62`, `v_1 = 100`, `t = 0.05`, and one mixed corner exists.
- Naive nominal fit uses two *different* values, e.g. top `160`, bottom `270`:
  nominal ratio `270/430 = 0.628`, but at the corner where the top drops and the
  bottom rises the ratio swings to `~0.65`, so `F ~ 0.03`.
- Matched fit: top `= 3` units of `v_1`, bottom `= 5` units of `v_1`, ratio
  `5/8 = 0.625` at **every** corner (the common batch error cancels), so `F = 0.005`
  from the rational-approximation gap alone. Spending more matched units shrinks that
  gap further. The insight is to make the ratio depend only on part *counts*, not on
  which absolute values drift.
```
Illustrative FORM only — your instance has many taps and corners; read them.
```
