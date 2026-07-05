# Debris Sweep Cordon: Additive-Basis Depot Placement

## Problem
A congested orbital shell is modelled as a ring of integer **phase slots**
`0, 1, ..., M`. You command a fleet of autonomous debris-sweeper depots and may
station at most `n` of them, each parked in a **distinct** integer phase slot in
`[0, M]`. Let `A` be your set of occupied slots.

A drifting debris fragment sitting at phase slot `t` can be **captured** if two of
your depots can jointly reach it with a single cooperative manoeuvre:

- a **rendezvous burn**: two depots at slots `a` and `b` phase together and meet at
  slot `a + b` (so every element of the sum set `A + A` is capturable), **or**
- a **phasing transfer**: a depot at slot `a` drifts down to meet one at slot `b`,
  reaching slot `a - b` when `a >= b` (so every non-negative element of the
  difference set `A - A` is capturable).

The mission needs a **contiguous cleared band** starting at slot `0`: you want the
largest `N` such that *every* slot `0, 1, ..., N` is capturable. Formally, let

```
R = (A + A)  U  { a - b : a, b in A, a >= b }
```

be the set of capturable slots. Your objective is to **maximise the reach**
`N = max { N : {0,1,...,N} ⊆ R }` (equivalently, the first uncovered slot minus one).

Because you have far fewer depots than slots, you cannot simply occupy a solid
block; you must place depots so their pairwise sums and differences **tile an
initial interval as far out as possible** — an additive-basis (postage-stamp)
packing problem whose exact optimum is unknown.

## Input (stdin)
One line with two integers:
```
n M
```
`n` = depot budget (maximum number of slots you may occupy), `M` = highest phase
slot. It is guaranteed that `M >= n - 1`.

## Output (stdout)
First line: an integer `k` (`1 <= k <= n`), the number of depots you actually place.
Then `k` integers (whitespace/newline separated), the occupied phase slots.

```
k
p_1
p_2
...
p_k
```

## Feasibility
- You must output between `1` and `n` slots (`1 <= k <= n`).
- Every slot must be an integer in `[0, M]`.
- All slots must be distinct.

Any violation scores `Ratio: 0.0`.

## Objective
Maximise the reach `F = N`, the length of the contiguous cleared band
`{0,1,...,N} ⊆ R` (larger is better).

## Scoring
The checker builds an internal baseline `B = 2n - 2`, the reach of the
arithmetic-progression cordon `0,1,...,n-1` (whose sum set is exactly the block
`[0, 2n-2]`). The score is
```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```
So matching the arithmetic-progression cordon scores `0.1`, and reaching `10x` the
baseline caps at `1.0`. The objective is graded and admits no easy optimum: within
the loose slot budget, even the best known additive basis reaches only a few times
the baseline, and the extremal reach is unknown.

## Constraints
- `6 <= n <= 200`, `M ~ 4n` across the test ladder.
- Deterministic exact scoring; no randomness or timing in the score.

## Example
Suppose `n = 3`, `M = 12`. The cordon `A = {0,1,2}` has
`R = {0,1,2,3,4}` (sums `0..4`), so `F = 4 = B` and `Ratio = 0.1`.
The layout `A = {0,1,3}` gives sums `{0,1,2,3,4,6}` and differences `{0,1,2,3}`,
so `R = {0,1,2,3,4,6}`; slot `5` is uncovered, hence `F = 4` as well.
The layout `A = {0,1,5}` gives sums `{0,1,2,5,6,10}` and differences `{0,1,4,5}`,
so `R = {0,1,2,4,5,6,10}`; slot `3` is uncovered, `F = 2`.
Choosing `A = {0,2,3}`: sums `{0,2,3,4,5,6}`, differences `{0,1,2,3}`, so
`R = {0,1,2,3,4,5,6}`, giving `F = 6` and
`Ratio = min(1000, 100*6/4)/1000 = 0.15`.
