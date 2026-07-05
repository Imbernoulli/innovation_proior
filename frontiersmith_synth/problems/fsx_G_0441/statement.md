# Minimum-Comparator Sorting Network

## Problem
A **comparator sorting network** on `n` wires is a fixed, data-independent sequence of
compare-exchange operations. A comparator `(i, j)` reads the values on wires `i` and `j`,
writes the **minimum** onto wire `i` and the **maximum** onto wire `j`. A network *sorts*
if, for every possible input assignment, the values on the wires end up in non-decreasing
order (wire `0` smallest, wire `n-1` largest).

Because comparators are monotone, by the **zero-one principle** a network sorts every
real input if and only if it sorts all `2^n` binary inputs. The checker uses exactly this
test.

Your job: for the given `n`, output a sorting network that uses **as few comparators as
possible**. This is a genuine open problem: the minimum comparator count is known to be
optimal only for `n <= 12`; for the values of `n` used here (`n >= 13`) the true optimum
is unknown, and standard merge-based constructions are far from tight.

## Input (stdin)
A single integer `n` — the number of wires (`13 <= n <= 22`).

## Output (stdout)
A list of comparators, one per line, each as two space-separated integers `i j` with
`0 <= i, j < n` and `i != j`. Comparator `(i, j)` places the minimum on wire `i` and the
maximum on wire `j`. Comparators are applied **in the order printed**. The number of
comparators is the number you print; there may be at most `2000` of them.

An empty output, malformed tokens, non-finite values, out-of-range indices, or a network
that fails to sort some input all score `0`.

## Feasibility
The output is feasible iff every token is an integer, indices are in range, no comparator
uses a single wire, there are at most `2000` comparators, and the resulting network sorts
all `2^n` binary inputs.

## Objective (minimize)
Let `F` be the number of comparators in your (correct) network.

## Scoring
The checker builds its own baseline `B = n*(n-1)/2` (the bubble-sort network). For a
feasible, correctly-sorting network it reports

```
sc    = min(1000, 100 * B / F)
Ratio = sc / 1000
```

So a bubble-cost network scores `0.1`; halving the comparator count roughly doubles the
score; the ratio only reaches `1.0` for a (physically unattainable) `~10x` reduction, so
there is always headroom above the merge-network constructions. Infeasible output scores
`Ratio: 0.0`.

## Constraints
- `13 <= n <= 22`.
- At most `2000` comparators.
- Scoring is exact and deterministic (parallel zero-one verification over all `2^n`
  inputs using big-integer columns); nothing is timed.

## Example
For `n = 4`, the network
```
0 1
2 3
0 2
1 3
1 2
```
uses `5` comparators and sorts all `16` binary inputs (this is in fact optimal for `n=4`).
Against a bubble baseline `B = 4*3/2 = 6`, it would score `min(1000, 100*6/5)/1000 = 0.12`.
(The graded instances all use `n >= 13`, where no optimum is known.)
