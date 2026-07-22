# Digit-Selection Carpet: Matching Roughness and Gaps

## Problem

You build a self-similar fractal ("digit-selection carpet") as a k-level
iterated function system on an n x n grid. At level `i` (i = 1..k) you pick a
**mask** `M_i`: a subset of the n x n grid cells, `1 <= |M_i| <= n*n - 1`
(a proper, non-empty subset -- never the full grid). The attractor is grown
to resolution `L = n^k` by nesting: cell `(x, y)` of the finest `L x L` grid
(written as its `k` base-`n` digit pairs `(x_1,y_1), ..., (x_k,y_k)`) is
**occupied** iff `(x_i, y_i) in M_i` for every level `i`.

Two statistics are measured on the grown attractor:

- **Box-counting dimension** `D`: let `N_j = |M_1|*|M_2|*...*|M_j|` be the
  number of occupied boxes of side `n^{-j}` (this count depends only on the
  *cardinalities* `|M_i|`, not on which cells they are). `D` is the
  least-squares slope of `ln(N_j)` against `j*ln(n)` for `j = 1..k`.
- **Lacunarity** `Lam`: slide a box of side `r = n^{floor(k/2)}` (clamped so
  it fits) over the finest `L x L` occupancy grid at every integer offset;
  let `mass(pos)` be the number of occupied finest cells inside the box at
  that position. `Lam = mean(mass^2) / mean(mass)^2` over all box positions
  -- the standard gliding-box heterogeneity index (>= 1; higher means the
  mass is lumped into fewer regions with bigger gaps elsewhere).

You are given a target dimension `D*` and target lacunarity `Lam*`. Your job
is to choose the `k` masks so the grown attractor's measured `(D, Lam)` is
close to `(D*, Lam*)`.

**The trap**: `D` depends only on how MANY cells each mask keeps, so hitting
`D*` alone is easy (pick cardinalities whose product gives the right slope)
and tells you nothing about `Lam*`. `Lam` depends only on WHICH cells are
kept -- the spatial arrangement. A construction that fixes cardinalities to
hit `D*` and then places cells however is convenient (e.g. always the same
compact block) will usually miss `Lam*` badly, because compact/clustered
placements and spread-out placements at the *same* cardinality produce very
different lacunarities. Matching both requires treating cardinality-choice
and cell-placement as separate decisions.

## Input (stdin)
```
n k
Dstar Lamstar
wD wL
```
`n` (2..6) is the grid base, `k` (>=3) is the number of levels, `Dstar` and
`Lamstar` are the targets, and `wD`, `wL` are the weights used by the scoring
formula below (read and use them -- they vary per test).

## Output (stdout)
```
k
N_1 r_1 c_1 r_2 c_2 ... r_{N_1} c_{N_1}
...
N_k r_1 c_1 ... r_{N_k} c_{N_k}
```
First print `k` (must equal the input `k`), then one line per level: the
cardinality `N_i` followed by `N_i` distinct cell coordinates
`0 <= r,c < n` for `M_i`.

## Feasibility
- The first token must equal the input `k`.
- Every `N_i` must satisfy `1 <= N_i <= n*n - 1`.
- The `N_i` coordinate pairs on a level must be distinct and within
  `[0, n-1] x [0, n-1]`.
- No missing or extra tokens; all values finite integers.
Any violation scores `Ratio: 0.0`.

## Objective (minimize)
`F = wD * |D_measured - Dstar| + wL * |ln(Lam_measured) - ln(Lamstar)|`
(lacunarity is matched on a log scale, standard for this heavy-tailed
statistic, so a sparse/clustered miss doesn't blow up the score).

## Scoring
The checker also builds its own naive reference (one fixed cardinality
`floor(n^Dstar)`, filled in row-major order at every level, ignoring
`Lam*`) to get a baseline distance `B`. Your score is
`min(1.0, 0.1 * B / F)` (printed as `Ratio: <value>` -- a construction that
achieves distance 10x smaller than the naive reference saturates at 1.0).

## Constraints
`2 <= n <= 6`, `3 <= k <= 6`, `L = n^k <= 1296`. Time limit 5s.

## Example (illustrative form only, not a real test case)
`n=2, k=2`: masks `M_1 = {(0,0),(1,1)}`, `M_2 = {(0,0),(1,1)}` give
`N_1=2, N_2=4`. `D` = slope through `(ln2, ln2)` and `(2ln2, ln4)` = 1.0
(a diagonal line, as expected). This illustrates the mechanics only; the
actual instances use larger `n,k` and non-trivial targets.
