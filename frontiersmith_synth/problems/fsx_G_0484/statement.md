# Anchor Words: Minimum-Size Binary Covering Codes for r-Fault Memory

## Problem
A fault-tolerant memory stores `n`-bit words. Because up to `r` bits can silently flip,
the controller keeps a small table of **anchor words** and, on read-back, snaps the observed
word to any anchor within Hamming distance `r`. For this to always succeed, **every** word in
`{0,1}^n` must lie within distance `r` of some anchor — i.e. the anchor set must be a
**binary covering code of radius `r`**. Each extra anchor costs table space and comparison
energy, so you want as **few** anchors as possible.

Your task: given `n` and `r`, output a covering code of radius `r` over `{0,1}^n` using few
codewords (anchors).

## Input (stdin)
One line: two integers
```
n r
```
with `1 <= r < n`. (`n <= 13` in the test ladder.)

## Output (stdout)
One codeword per line. Each codeword is a string of exactly `n` characters over `{0,1}`
(bit `i` is the value of coordinate `i`, most-significant first). Print at least one line.
Duplicate lines are collapsed to a set before scoring.

Let `C` be the set of distinct codewords you print.

## Feasibility
`C` is valid iff it is a covering code of radius `r`:
for every `x` in `{0,1}^n` there exists `c` in `C` with `Hamming(x, c) <= r`.
Any malformed line (wrong length, a character other than `0`/`1`), an empty set, or an
uncovered word makes the whole submission infeasible.

## Objective
**Minimize** `F = |C|`, the number of distinct anchor words.

## Scoring
Deterministic. The checker builds its own natural-order **first-fit** covering code and lets
`B` be its size (an internal baseline). With your feasible size `F`:
```
sc    = min(1000, 100 * B / F)
Ratio = sc / 1000
```
Reproducing the baseline scores `Ratio ~= 0.1`; a code ten times smaller caps at `1.0`.
Any feasibility violation scores `Ratio: 0.0`. The true minimum (the covering number
`K(n, r)`) is unknown for these sizes, so there is genuine headroom above the baseline and
many viable strategies (first-fit orderings, look-ahead greedy, linear/coset codes,
annealing, ILP).

## Constraints
- `1 <= r < n`, `n <= 13`.
- Purely combinatorial; scoring uses exact integer arithmetic only.

## Example (worked score)
For a tiny instance, suppose the checker's first-fit baseline uses `B = 30` anchors and you
submit a valid covering code with `F = 15` distinct anchors. Then
`sc = min(1000, 100 * 30 / 15) = 200` and `Ratio = 0.200`. Halving the anchor count over the
baseline doubles the score. (Numbers illustrative only — actual `B` is computed per instance.)
