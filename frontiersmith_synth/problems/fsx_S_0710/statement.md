# Firebreak: Rescue Cash in an Interbank Clearing Network

## Problem
`N` banks owe each other money. Bank `i`'s **total obligation** `p_bar_i` is the sum
of everything it owes to other banks. Bank `i` also holds `e_i` in outside cash.
When settlement happens, every bank pays **as much as it can, pro-rata across its
own creditors**: if bank `i` cannot cover `p_bar_i` in full, each creditor of `i`
receives only its proportional share of what `i` actually manages to pay.

Because a bank's incoming payments depend on what its own debtors pay (which
depends on what *their* debtors pay, ...), the actual payment vector `p` is the
**greatest fixed point** of
```
p_i = min( p_bar_i,  e_i + sum_j ( L[j][i] / p_bar_j ) * p_j )     for every bank i with p_bar_i > 0
```
(`L[j][i]` = amount bank `j` owes bank `i`; a bank with `p_bar_i = 0` never owes
anyone and always pays `p_i = 0`). This is the classical Eisenberg-Noe clearing
model. A bank **defaults** iff `p_i < p_bar_i`; the **system-wide shortfall** is
`sum_i (p_bar_i - p_i)`.

You control a rescue budget `C`: cash you may hand out as external assets to any
banks *before* settlement happens. Your job is to choose the injection that
leaves the smallest possible system-wide shortfall after the (now-augmented)
network clears. A dollar's value to the system is **not** what it does for the
bank that receives it — it is whatever that dollar changes about the whole
fixed point once every other bank's payments re-equilibrate around it.

## Input (stdin)
```
N C
e_1 e_2 ... e_N
M
u_1 v_1 w_1
...
u_M v_M w_M
```
`N` banks (1-indexed `1..N`), rescue budget `C` (integer). `e_i` = bank `i`'s
outside cash (integer >= 0). Then `M` distinct directed liabilities: bank `u_k`
owes bank `v_k` amount `w_k` (`u_k != v_k`, `w_k >= 1`, integer). `p_bar_i` is
the sum of `w_k` over edges with `u_k = i`.

## Output (stdout)
```
delta_1 delta_2 ... delta_N
```
`N` numbers (any real, printed with reasonable precision): `delta_i >= 0` is the
cash injected into bank `i`, i.e. bank `i`'s external assets become `e_i +
delta_i`.

## Feasibility
Invalid (scores `Ratio: 0.0`) if: the output does not contain exactly `N`
parseable, finite numbers; any `delta_i < 0`; or `sum(delta_i) > C`.

## Objective
Let `p*` be the clearing payment vector computed with augmented assets
`e_i + delta_i`. Minimize `F = sum_i (p_bar_i - p*_i)`, the total post-rescue
shortfall.

## Scoring
The checker computes its own trivial reference `B`: the system-wide shortfall
when **no cash is injected at all** (`delta = 0`). With minimization
normalization,
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Doing nothing reproduces `B` and scores `Ratio = 0.1`. Cutting the shortfall to
a tenth of `B` caps the score at `1.0`.

## Constraints
- `1 <= N <= 400`, `0 <= M <= 3000`, `1 <= w_k <= 10^4`, `0 <= C, e_i <= 10^6`.
- Time limit 5s, memory 512MB.

## Example
Two banks: bank `1` owes bank `2` amount `10`; bank `1` has `e_1 = 4`, bank `2`
has `e_2 = 20` and owes nobody. Input:
```
2 6
4 20
1
1 2 10
```
With no injection, bank 1 pays `min(10, 4) = 4`, so `p_bar_1 - p*_1 = 6 = B`.
Injecting the whole budget into bank 1 (`delta = [6, 0]`) gives `e'_1 = 10`, so
bank 1 pays `10` in full: `F = 0`, `sc = min(1000, 100*6/1e-9)` capped at
`1000`, `Ratio = 1.0`. (This tiny illustrative instance has only one bank worth
funding; real instances hide the profitable target inside a network where the
biggest individual shortfall is usually a decoy.)
