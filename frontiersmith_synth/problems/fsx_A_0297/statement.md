# Harbor Counterweight Manifest — Maximizing Balance Diversity

## Problem

A container terminal must assign **distinct integer weight classes** (in tonnes) to a fleet
of `n` containers. The gantry cranes exploit two derived quantities of the chosen weight
multiset `A = {a_1, ..., a_n}` (all distinct):

- The **lift signature** is the set of all *combined* weights that a tandem lift can present,
  `A + A = { a_i + a_j : i <= j }`. Duplicated combined weights waste distinct crane presets.
- The **balance signature** is the set of all *counterweight differences* used to trim the
  gantry, `A - A = { a_i - a_j : all i, j }`. A larger balance signature means finer,
  more diverse trimming.

The port wants a manifest whose *balance diversity* is as rich as possible **relative to** the
lift signature it is forced to carry. Formally, maximize the exact rational

```
    quality(A) = |A - A| / |A + A|.
```

An arithmetic-progression manifest (consecutive weights) achieves `quality = 1`. Doing much
better is genuinely hard: pushing all pairwise differences apart tends to also spread the
pairwise sums. There is **no known optimal construction**; Sidon-type packings reach roughly
`2`, and whether one can do substantially better is open.

Some weight classes are **reserved** (already occupied berth codes) and may not be used.

## Input (stdin)

```
n M k
f_1 f_2 ... f_k
```

- `n` — the exact number of distinct weight classes to output (the manifest size).
- `M` — the maximum allowed weight class; every chosen weight is an integer in `[0, M]`.
- `k` — the number of reserved (forbidden) weight classes.
- `f_1 ... f_k` — the reserved weight classes (distinct integers in `[0, M]`). If `k = 0`
  this line is empty.

## Output (stdout)

Print exactly `n` **distinct** integers in `[0, M]`, none of them reserved, separated by
whitespace (spaces and/or newlines). This is your manifest `A`.

## Feasibility

An output is feasible iff it contains **exactly `n` tokens**, each a finite integer in
`[0, M]`, all **distinct**, and **none reserved**. Any violation scores `0`.

## Objective

Maximize `quality(A) = |A - A| / |A + A|`, computed with exact integer arithmetic over the
sumset and difference set.

## Scoring

Let `F = quality(A)` for a feasible manifest, and let `B` be the quality of the checker's
internal baseline (the first `n` non-reserved integers `0, 1, 2, ...`, i.e. a near
arithmetic progression, `B ≈ 1`). The reported score is

```
    Ratio = min(1000, 100 * F / B) / 1000.
```

A near-AP manifest scores about `0.1`; a manifest ten times richer than the baseline caps at
`1.0`. Infeasible or non-finite output scores `0`.

## Constraints

- `12 <= n <= 200`, `0 <= M`, `0 <= k <= n`, reserved classes distinct and in `[0, M]`.
- The checker runs in `O(n^2)` exact integer arithmetic; scoring is fully deterministic.

## Example (worked score)

Suppose `n = 4`, `M = 20`, `k = 0`.

- Baseline manifest `{0,1,2,3}`: `A+A = {0,1,2,3,4,5,6}` (7 values),
  `A-A = {-3,-2,-1,0,1,2,3}` (7 values), so `B = 7/7 = 1`.
- A Sidon-style manifest `{0,1,4,9}`: `A+A = {0,1,2,4,5,8,9,10,13,18}` (10 values),
  `A-A = {-9,-8,-5,-4,-3,-1,0,1,3,4,5,8,9}` (13 values), so `F = 13/10 = 1.3`.
- Score `= min(1000, 100 * 1.3 / 1.0) / 1000 = 0.13`.
