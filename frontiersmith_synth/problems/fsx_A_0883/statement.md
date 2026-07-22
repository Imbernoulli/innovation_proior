# Custom-Order BWT: Minimize Runs by Reordering and Rotating

## Problem
The Burrows-Wheeler Transform (BWT) sorts the cyclic rotations of a string and reads off
the last column. Textbook BWT always sorts symbols by their natural numeric order and
always cuts the cyclic string at a fixed position. Neither choice is forced: you may pick
**any total order on the alphabet** (which symbol counts as "smaller") and **any rotation**
of the string before the transform is applied. Both choices change which rotations end up
adjacent after sorting, and therefore change how many maximal equal-symbol runs appear in
the output. Your job: choose an alphabet order and a rotation that make the transform's
output as run-compressed as possible.

You are given a string `T` of length `n` over the alphabet `{0, 1, ..., k-1}`.

## Input (stdin)
```
n k
T[0] T[1] ... T[n-1]
```
`1 <= n <= 2000`, `2 <= k <= 200`, each `T[i]` an integer in `[0, k-1]`.

## Output (stdout)
```
p[0] p[1] ... p[k-1]
r
```
`p` must be a permutation of `0..k-1`: `p[i]` is the symbol assigned rank `i` (rank 0 is
"smallest"). `r` is the rotation offset, `0 <= r < n`.

## Construction (how the checker scores you)
Let `U = T[r], T[r+1], ..., T[n-1], T[0], ..., T[r-1]` (the rotation of `T` by `r`). Append
one sentinel symbol `$`, defined to rank strictly below every symbol in `p` regardless of
`p`. Form all `n+1` cyclic rotations of `U + $`, sort them lexicographically using the
symbol order `p` (with `$` smallest), and let `BWT` be the resulting last column (length
`n+1`, read from the sorted rotation list top to bottom). The **objective** is the number
of maximal runs of equal symbols in `BWT` (adjacent equal characters count as one run;
`$` occurs once and always starts/ends its own run).

## Feasibility
Output is feasible iff: exactly `k+1` whitespace-separated tokens are given, the first `k`
parse as integers forming a permutation of `0..k-1`, and the last parses as an integer with
`0 <= r < n`. `nan`/`inf`/non-integer tokens, wrong token count, a non-permutation, or an
out-of-range `r` all score `0`.

## Objective
Minimize `runs(p, r)`, the run count defined above.

## Scoring
Let `B = n+1` — the trivial upper bound on the run count (a length-`(n+1)` `BWT` column
trivially cannot contain more than `n+1` maximal runs). With `F = runs(p, r)` your
submission achieves,
```
Ratio = min(1, 0.1 * B / F)
```
The textbook default (`p` = identity, `r = 0`) already compresses well below the trivial
upper bound, so it scores modestly above `0.1`, not exactly `0.1` — reproducing it is a
weak baseline, not a competitive one. Halving `F` roughly doubles the ratio; reaching a
tenth of `B` caps the score at `1.0`. The order that truly minimizes runs is not
efficiently computable in general (finding the alphabet permutation minimizing BWT run
count is a hard combinatorial ordering problem), so headroom remains above what any of
the reference strategies below reach.

## Constraints
- `1 <= n <= 2000`, `2 <= k <= 200`.
- Time limit: 5 seconds. Memory: 512 MB.
- Deterministic scoring; the checker never times anything.

## Example
Suppose `n=6, k=3, T = 0 1 0 2 1 0`, so `B = n+1 = 7`. With `p=(0,1,2)` (identity) and
`r=0`: rotating gives `U=T`, and sorting the 7 rotations of `T+$` under natural order
yields a `BWT` with `F=6` runs, scoring `min(1, 0.1*7/6) = 0.1167`. A submission finding
`p=(0,2,1), r=2` instead reaches `F=4`, scoring `min(1, 0.1*7/4) = 0.175`. These example
numbers are exact for this tiny `T`; `B` is recomputed from `n` for every input.
