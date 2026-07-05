# Warehouse Line-Free Shelf Activation (Weighted Cap Set in F_3^n)

## Problem
A cubic warehouse stores goods on shelves indexed by the ternary grid `F_3^n`
(all length-`n` vectors over `{0,1,2}`). A robot fleet can *activate* a subset of
shelves. Three distinct shelves `a,b,c` are said to lie on a **hazard line** when

```
a[k] + b[k] + c[k] ≡ 0 (mod 3)   for every coordinate k = 0..n-1
```

(equivalently `c` is the unique third point of the affine line through `a` and `b`
in the geometry `AG(n,3)`). If all three shelves of any hazard line are active, the
robots' routing loops become degenerate and collide. The set of active shelves must
therefore be **line-free**: it must contain no hazard line at all — this is exactly a
*cap set* in `F_3^n`.

Each shelf `p` has a positive throughput weight `w(p)`. Activate a line-free set of
shelves that maximises total throughput.

## Input (stdin)
```
n
w_0 w_1 ... w_{N-1}
```
- Line 1: the dimension `n` (`4 ≤ n ≤ 7`).
- Then `N = 3^n` positive integers, whitespace-separated (possibly across many lines):
  `w_i` is the throughput weight of the shelf whose index is `i`.
- **Index encoding.** Shelf index `i` in `[0, N)` decodes to the vector
  `x` with `x[k] = floor(i / 3^k) mod 3` (little-endian base-3). Equivalently
  `i = sum_k x[k] * 3^k`.

## Output (stdout)
```
k
i_1 i_2 ... i_k
```
- `k` = number of activated shelves, then the `k` distinct shelf **indices**
  (each in `[0, N)`), whitespace-separated (line breaks allowed). `k` may be 0.

## Feasibility
The output is feasible iff:
- every listed index is an integer in `[0, N)`,
- indices are pairwise distinct,
- the activated set contains **no hazard line** (no three distinct active shelves
  `a,b,c` with `a[k]+b[k]+c[k] ≡ 0 (mod 3)` in every coordinate).

Any violation (bad token, out-of-range/duplicate index, non-finite value, or a hazard
line present) scores `Ratio: 0.0`.

## Objective (maximize)
`F = sum of w(p) over all activated shelves p`.

## Scoring
Let `B` be the total weight of the **binary sub-cube baseline** `{0,1}^n` (all shelves
whose every coordinate is 0 or 1 — always line-free, since any hazard line needs a
coordinate hitting all of `{0,1,2}`). The score is
```
Ratio = min(1000, 100 * F / B) / 1000
```
Reproducing the baseline scores ≈ 0.10; a fully weight-optimised line-free set scores
higher. The maximum-weight cap set is not known in closed form for these `n`, so the
objective is genuinely open-ended: many distinct activation heuristics are viable.

## Constraints
- `4 ≤ n ≤ 7`, `N = 3^n` (81, 243, 729, 2187). Weights are integers in `[1, 10^6]`.
- Deterministic scoring; exact integer arithmetic only.

## Example (worked score — illustrative shape only, NOT the graded instance)
Suppose `n = 4` and the baseline sub-cube weight is `B = 9426`. If your activated
line-free set has total weight `F = 14063`, then
`Ratio = min(1000, 100 * 14063 / 9426)/1000 = min(1000, 149.19)/1000 = 0.149`.
A set equal to the sub-cube scores `100*9426/9426/1000 = 0.100`.
