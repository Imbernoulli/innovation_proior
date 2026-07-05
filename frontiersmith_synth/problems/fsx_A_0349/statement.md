# Coral Reef Survey: Minimum Star-Discrepancy Station Placement

## Problem

A marine research team is planning a benthic survey of a coral reef. The reef has been
rectified onto the unit square `[0,1]^2`. A survey needs `M` sampling stations spread so that
*every* rectangular sub-region anchored at the reef's south-west corner is sampled in
proportion to its area -- otherwise some habitats are over- or under-counted. The mathematical
measure of this uniformity is the **star discrepancy** of the station set (lower is better).

`K` stations are already fixed (pre-installed moored buoys you cannot move). Your job is to
choose the coordinates of the remaining `A = M - K` stations so that the star discrepancy of
the **full** set (fixed buoys together with your new stations) is as small as possible.

## Input (stdin)

```
d M K
x_1 y_1
...
x_K y_K
```

- `d = 2` (the reef is a 2-D unit square).
- `M` = total number of stations desired.
- `K` = number of fixed stations, followed by their `K` coordinate lines, each in `[0,1]^2`.

## Output (stdout)

Exactly `A = M - K` lines, each `x y`, the coordinates of your added stations, with
`0 <= x <= 1` and `0 <= y <= 1`. Values may be given as decimals. Duplicate points are
permitted but rarely help.

## Feasibility

The output must contain exactly `A` finite coordinate pairs, each inside `[0,1]^2`. Any other
count, non-finite (`nan`/`inf`), out-of-range, or unparseable value makes the submission
infeasible and scores 0.

## Objective (minimize)

Let `S` be the full set of `n = M` stations (the `K` fixed buoys plus your `A` stations). The
**star discrepancy** is

```
D*(S) = sup over (a,b) in [0,1]^2 of  | #{ p in S : p_x <= a and p_y <= b } / n  -  a*b |.
```

The supremum is attained on the finite grid formed by the station coordinates, so it is
computed **exactly** (evaluating both the closed-box count-heavy and open-box volume-heavy
cases at every grid corner). You want `D*(S)` as small as possible.

## Scoring

The checker builds an internal baseline `B` = the star discrepancy of the trivial layout that
dumps all `A` added stations at the reef centre `(0.5, 0.5)`. With your discrepancy `F`:

```
sc = min(1000, 100 * B / F)
Ratio = sc / 1000
```

Reproducing the trivial layout scores about `0.1`; halving the discrepancy roughly doubles the
score; a set an order of magnitude better than the baseline saturates at `1.0`. There is no
known optimal construction for a general `n` in 2-D, so the problem stays genuinely open-ended:
Halton and Sobol nets, Fibonacci / rank-1 lattices, and greedy discrepancy minimization are all
viable and beat each other on different instances.

## Constraints

- `2 <= K < M <= 60`, `d = 2`.
- Time limit 5 s, memory 512 MB.

## Example

Suppose the input is `2 4 2` with fixed buoys `(0.10, 0.80)` and `(0.90, 0.20)`, so you must
add `A = 2` stations. Emitting

```
0.25 0.25
0.75 0.75
```

places one station in the south-west quadrant and one in the north-east quadrant. The full
4-point set is fairly spread out, giving a small star discrepancy and a Ratio well above the
`0.1` you would get by placing both added stations at `(0.5, 0.5)`. (Illustrative only -- the
scored instances are larger.)
