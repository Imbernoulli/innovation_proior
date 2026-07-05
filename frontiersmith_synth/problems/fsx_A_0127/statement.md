# Metro Ladder: Maximizing Combined Through-Fares and Skip-Gaps

## Problem
A new subway line is a single straight track. You may build up to `k` **stations**,
each at a **distinct** integer **milepost** drawn from `[0, V]`. Let `A` be the set of
mileposts you choose.

Riders derive value from the layout in two independent ways:

- **Through-fares.** A through-service train chains **exactly two stations** (the two may
  be the **same** station, boarded twice). The fare-endpoint is the **sum** of the two
  mileposts. The set of achievable fare-endpoints is the **sumset**
  `A + A = { a + b : a, b in A }` (doubles `a + a` count). More distinct fare-endpoints
  means more distinct destinations the through network can price.

- **Skip-gaps.** An express train that skips from station `a` to station `b` traverses the
  **signed gap** `a - b`. The set of achievable skip-gaps is the **difference set**
  `A - A = { a - b : a, b in A }` (it always contains `0`, and is symmetric: if `d` is
  achievable so is `-d`). More distinct skip-gaps means more distinct express patterns.

You want the line to offer as much distinct service as possible **overall**. Your objective
is to **maximize the total number of distinct service values**

```
F = |A + A| + |A - A|.
```

Both quantities are largest when the mileposts avoid additive coincidences (a **Sidon /
B_2 layout**: all pairwise sums distinct, equivalently all pairwise differences distinct).
But the mileposts must all fit inside the bounded corridor `[0, V]`, so when `V` is tight
you cannot spread the stations out freely and a perfect Sidon layout of size `k` may be
impossible — that is where the design gets hard.

## Input (stdin)
```
k V
```
- `k` — the maximum number of stations you may build (the size budget).
- `V` — the inclusive upper bound on a milepost; every milepost is an integer in `[0, V]`.

## Output (stdout)
```
m
<m lines, each one integer = one station milepost>
```
Print the number of stations you build `m` (with `0 <= m <= k`), then the `m` mileposts,
one per line.

## Feasibility
An output is valid iff **all** hold:
- `0 <= m <= k`;
- exactly `m` integers follow;
- every milepost is an integer with `0 <= milepost <= V`;
- the mileposts are **pairwise distinct**.

Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = |A + A| + |A - A|`. Both the sumset and the difference set are computed
**exactly** with integer arithmetic (doubles `a + a` and the zero difference `a - a` are
included).

## Scoring
Let `B` be the value `F` of the checker's own trivial construction: the **contiguous ladder**
`{0, 1, ..., m0 - 1}` with `m0 = min(k, V + 1)` stations. For the ladder
`A + A = {0, ..., 2*m0 - 2}` (size `2*m0 - 1`) and `A - A = {-(m0-1), ..., m0-1}` (size
`2*m0 - 1`), so `B = 2*(2*m0 - 1) = 4*m0 - 2`.

With maximization normalization:
```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the ladder scores `Ratio = 0.1`; a layout offering `10x` more distinct service
than the ladder caps at `1.0`.

## Constraints
- `16 <= k <= 48`.
- `k <= V <= 2 * k * k` (the corridor is always wide enough for the ladder baseline, and
  sometimes too tight for a perfect Sidon layout of size `k` — that is where the problem
  gets hard).
- Time limit 5s, memory 512m.

## Example
Let `k = 4`, `V = 20`. The ladder `{0,1,2,3}` has `A+A = {0..6}` (7 values) and
`A-A = {-3..3}` (7 values), so `B = 14` and reproducing it scores `0.1`.
The Sidon layout `{0,1,4,9}` has `A+A = {0,1,2,4,5,8,9,10,13,18}` (10 values) and
`A-A = {-9,-8,-5,-4,-3,-1,0,1,3,4,5,8,9}` (13 values), giving `F = 23`,
`sc = 100 * 23 / 14 = 164.2857`, `Ratio = 0.164286`.
