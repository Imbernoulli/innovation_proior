# Unambiguous Coincidence Signatures for a Volcano Sensor Network

## Problem
A volcano-monitoring network deploys a set of seismic sensors around the crater.
Each sensor is assigned an integer **firing offset** (a fixed processing delay,
in milliseconds) drawn from the range `[0, M]`. When a tremor occurs, any pair of
sensors `{i, j}` reports a combined **coincidence signature** equal to the sum of
their two offsets, `off_i + off_j` (a sensor may also pair with itself). The
central station distinguishes tremor micro-events only when their coincidence
signatures differ, so signatures that collide create ambiguous, wasted readings.

You may deploy **at most `n` sensors**, each with a **distinct** integer offset in
`[0, M]`. Choose the offsets so that the number of **distinct coincidence
signatures** is as large as possible. If `A` is your multiset-free set of offsets,
you are maximizing the size of the sumset

```
A + A = { x + y : x in A, y in A }.
```

## Input (stdin)
One line with two integers:
```
n M
```
`n` is the sensor budget (`|A| <= n`); `M` is the maximum offset value.

## Output (stdout)
The chosen offsets, as whitespace-separated integers (one line is fine):
```
a_1 a_2 ... a_k
```
with `1 <= k <= n`.

## Feasibility
Your output is rejected (score 0) unless:
- every token is an integer,
- `1 <= k <= n`,
- all offsets are pairwise distinct,
- every offset lies in `[0, M]`.

## Objective
Maximize `F = |A + A|`, the number of distinct pairwise sums (including doubles
`x + x`).

## Scoring
Let `B = 2n - 1`, the sumset size of the dense baseline deployment
`{0, 1, ..., n-1}` (an arithmetic progression, which has the *fewest* distinct
sums). The score is

```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```

so the arithmetic-progression baseline scores about `0.1`, and reaching ten times
the baseline caps the ratio at `1.0`. Scoring is exact integer arithmetic and
deterministic.

## Constraints
- `15 <= n <= 42`, `4n <= M <= 9n` (small-scale ladder; offsets fit in machine
  integers).
- Because `M` is far below `n^2`, a perfect Sidon set (all sums distinct) of size
  `n` does not fit — you must trade off spread against budget usage. There is no
  known simple optimum.

## Example
Input:
```
4 20
```
The baseline `{0,1,2,3}` gives `A+A = {0,1,2,3,4,5,6}`, so `F = 7 = 2n-1 = B`,
Ratio `= 0.1`. The spread-out set `{0, 1, 4, 9}` gives sums
`{0,1,2,4,5,8,9,10,13,18}`, so `F = 10`, and Ratio `= min(1000, 100*10/7)/1000
= 0.142857`.
