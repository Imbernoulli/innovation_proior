# Telescope Array: Non-Degenerate Antenna Configurations

## Problem
You are laying out a radio-telescope array on a discrete `n`-dimensional
configuration lattice over the field `F_3 = {0, 1, 2}`. Each candidate antenna
site is a coordinate vector `x` in `F_3^n` (so there are `3^n` possible sites).

A triple of three **distinct** antennas `x, y, z` forms a **degenerate baseline
triple** when, in every coordinate `k`,

```
(x[k] + y[k] + z[k]) mod 3 == 0 .
```

(Equivalently, the three sites are affinely collinear in the geometry `AG(n, 3)`.)
Degenerate triples produce redundant / self-cancelling interferometric baselines
and must be avoided entirely.

Place as many antennas as possible so that **no** degenerate baseline triple
exists among the chosen sites. In combinatorial terms you are asked to construct
a large **cap set** in `F_3^n`. There is no known simple formula for the maximum,
and many different ordering / product heuristics give very different results —
this is a genuinely open-ended construction task.

## Input (stdin)
A single line with one integer `n` — the dimension of the configuration lattice.

```
n
```

Constraints: `3 <= n <= 8`.

## Output (stdout)
Print your chosen antenna sites, one per line. Each line must contain exactly
`n` integers, each in `{0, 1, 2}`, separated by single spaces — the coordinate
vector of one antenna. Blank lines are ignored. The number of lines is the size
of your configuration. You may output them in any order; duplicates are not
allowed.

```
x1_1 x1_2 ... x1_n
x2_1 x2_2 ... x2_n
...
```

## Feasibility
An output is **feasible** iff:
1. every line has exactly `n` entries, each in `{0, 1, 2}`;
2. all antenna vectors are pairwise distinct;
3. no three distinct chosen antennas form a degenerate baseline triple.

Any violation scores `0`.

## Objective
Maximize `F` = the number of antennas placed (the cardinality of your cap set).

## Scoring
Let `B` be the size of the checker's built-in trivial construction (the
"diagonal" configuration `{0, e_1, ..., e_n}`, of size `n + 1`). With `F` your
feasible cardinality, the score is

```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```

So reproducing the diagonal baseline scores about `0.1`, and reaching `10 * B`
antennas caps the ratio at `1.0`.

## Constraints
- `3 <= n <= 8`.
- Deterministic scoring; exact integer arithmetic.
- Time limit 5 s, memory 512 MB.

## Example
For `n = 3`, the diagonal configuration
```
0 0 0
1 0 0
0 1 0
0 0 1
```
has `F = 4 = B`, scoring `Ratio = 0.1`. A `9`-antenna cap set (the maximum for
`n = 3`) scores `Ratio = 0.1 * 9 / 4 = 0.225`.
