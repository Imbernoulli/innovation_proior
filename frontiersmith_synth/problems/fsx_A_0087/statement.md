# Echo Promenade: Distinct-Distance Stage Layout

## Problem
A festival promenade is a straight line of integer positions `0, 1, ..., M`.
You must place exactly `n` stages at **distinct** integer positions along it.
When two stages play at the same time, the sound reflected between them arrives
with a delay proportional to their separation. To keep every echo distinguishable
you want the multiset of pairwise separations to realize as **many distinct values
as possible**.

Formally, let `A` be your set of `n` chosen positions. Define the difference set
`A - A = { a - b : a, b in A }` (a symmetric set containing `0`). Your goal is to
**maximize `|A - A|`**, the number of distinct signed spacings. Equivalently, you
maximize the number of distinct *positive* pairwise distances (since
`|A-A| = 2 * (distinct positive distances) + 1`).

Because the promenade length `M` is tight (roughly `0.6 * C(n,2)`), a "perfect"
layout in which all `C(n,2)` distances are distinct (a Golomb ruler) does **not**
fit. You must pack distinct distances as densely as possible within `[0, M]`.

## Input (stdin)
One line with two integers:
```
n M
```
`n` = number of stages to place, `M` = maximum promenade position. It is guaranteed
that `M >= n - 1`.

## Output (stdout)
First line: an integer `k` (the number of stages you place, which must equal `n`).
Then `k` integers (whitespace/newline separated), the stage positions.

```
k
p_1
p_2
...
p_k
```

## Feasibility
- Exactly `n` positions must be output (`k == n`).
- Every position must be an integer in `[0, M]`.
- All positions must be distinct.

Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = |A - A|` (larger is better).

## Scoring
The checker builds an internal baseline `B = 2n - 1` (the difference-set size of the
arithmetic progression `0,1,...,n-1`). The score is
```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```
So matching the arithmetic-progression baseline scores `0.1`, and reaching `10x`
the baseline caps at `1.0`. The objective is graded and admits no easy optimum:
covering all distances in a tight range is a hard packing problem.

## Constraints
- `6 <= n <= 32`, `M <= 300` across the test ladder.
- Deterministic exact scoring; no randomness or timing in the score.

## Example
Suppose `n = 4`, `M = 5`. The arithmetic progression `A = {0,1,2,3}` has
`A - A = {-3,-2,-1,0,1,2,3}`, so `F = 7 = B` and `Ratio = 0.1`.
The layout `A = {0,1,4,5}` has positive distances `{1,3,4,5}` (four distinct),
so `F = 2*4 + 1 = 9`, giving `Ratio = min(1000, 100*9/7)/1000 = 0.128571`.
