# Conflict-Free Additive Codes (Maximum Sidon Set)

## Problem
A *conflict-free additive code* over the channel alphabet `{1, 2, ..., n}` is a set of
codewords (distinct integers) whose **pairwise sums never collide**: for every two
(unordered, with-repetition) pairs of codewords `{a, b}` and `{c, d}`,
`a + b = c + d` implies `{a, b} = {c, d}`. Equivalently, all sums `a + b` with
`a <= b` taken over the chosen set are pairwise distinct. Such a set is classically
called a **Sidon set** (a `B_2` set / additive perfect-difference code): it is exactly
the structure a receiver needs to unambiguously demodulate the superposition of any two
transmitters.

Given `n`, output as large a conflict-free code inside `{1, ..., n}` as you can.

## Input (stdin)
A single integer `n` (`n >= 6`).

## Output (stdout)
The chosen codewords: distinct integers in `[1, n]`, separated by whitespace (spaces
and/or newlines), in any order. Print **only** the codewords — no count, no other text.
The empty set is allowed but scores 0.

## Feasibility
An output is feasible iff:
1. every token is an integer in `[1, n]`;
2. the integers are pairwise distinct;
3. all pairwise sums `a + b` (over `a <= b` in the set) are pairwise distinct
   (the Sidon / conflict-free condition).

Any violation (out-of-range value, duplicate, sum collision, non-integer, non-finite
token) makes the output infeasible and scores 0.

## Objective (maximize)
The score grows with the **cardinality** `F` of the submitted conflict-free code.

## Scoring
Deterministic. Let `B` be the size of the checker's internal baseline code (the
powers of two `1, 2, 4, ...` that are `<= n`, which always form a conflict-free code).
The reported ratio is

```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```

Reproducing the baseline scores `Ratio = 0.1`; a code ten times larger caps at `1.0`.
The maximum Sidon set size is `~sqrt(n)`, but no exact optimal construction is known for
general `n`, so headroom is genuine.

## Constraints
- `6 <= n <= 8000` across the test ladder.
- Feasibility check is `O(F^2)`; keep `F` at most a few hundred.

## Example
For `n = 12`, the set `{1, 2, 5, 11}` has pairwise sums
`2, 3, 6, 12, 4, 7, 13, 10, 16, 22` — all distinct — so it is feasible with `F = 4`.
The baseline powers of two `{1, 2, 4, 8}` give `B = 4`, so this submission would score
`Ratio = min(1000, 100*4/4)/1000 = 0.1`. A larger valid set such as `{1, 2, 5, 11, 12}`
(if it stays conflict-free) would score higher.
