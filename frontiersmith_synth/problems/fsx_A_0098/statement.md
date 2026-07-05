# Reef Survey Response Tensor: Minimal Separable Survey Modes

## Problem
A coral-reef monitoring program summarizes a season of dive surveys as an integer
**response tensor** `S[i][j][k]`, where

- `i` indexes a **transect line** (`0 <= i < I`),
- `j` indexes a **depth band** (`0 <= j < J`),
- `k` indexes a surveyed **taxon** (`0 <= k < K`).

The entry `S[i][j][k]` is the (signed, integer) standardized abundance index for that
combination. To model the data compactly the program approximates it as a sum of
**separable survey modes**. A single mode `r` is a rank-1 pattern given by three
profiles — a transect profile `u_r` (length `I`), a depth profile `v_r` (length `J`)
and a taxon profile `w_r` (length `K`) — contributing

```
u_r[i] * v_r[j] * w_r[k]
```

to every cell `(i,j,k)`. Fitting one mode requires exactly **one scalar multiplier per
cell family**, so the reporting cost is the *number of modes* `R`. Your task: reproduce
the survey tensor **exactly** with **as few modes as possible** — i.e. find a low-rank CP
(canonical polyadic) decomposition of `S`.

## Input (stdin)
```
I J K
```
then `I*J` lines. The line at position `i*J + j` (0-based, `i` outer, `j` inner) lists the
`K` integers `S[i][j][0] S[i][j][1] ... S[i][j][K-1]`.

Dimensions satisfy `1 <= I, J, K <= 5`.

## Output (stdout)
```
R
```
followed by `R` modes. Each mode is written as `I + J + K` rational numbers (whitespace
separated; you may wrap lines freely):

```
u[0] ... u[I-1]  v[0] ... v[J-1]  w[0] ... w[K-1]
```

Each number may be an integer (`-3`), a decimal (`0.5`, `-1.25`) or an exact fraction
(`7/4`, `-3/2`). Scientific notation (`1e3`) and the tokens `nan` / `inf` are **rejected**.

## Feasibility
Let `Shat[i][j][k] = sum_{r=1}^{R} u_r[i] * v_r[j] * w_r[k]`, evaluated in exact rational
arithmetic. The output is feasible iff `Shat == S` **exactly** at every cell. Any parse
error, wrong token count, non-finite value, `R < 1`, or reconstruction mismatch scores 0.

## Objective
Minimize `R`, the number of separable survey modes.

## Scoring
Let `B` be the number of nonzero cells of `S` (the naive "one mode per nonzero cell"
construction always achieves rank `B`). With your feasible mode count `R`,

```
Ratio = min(1, 0.1 * B / R)
```

The per-cell baseline scores `0.1`. Fewer modes score higher; reaching a tenth of `B`
caps at `1.0`. The survey tensor has a genuinely **unknown** minimal rank — it is planted
with a low *multilinear* (Tucker) rank but its true CP rank lies well above the largest
axis, so no polynomial method recovers the optimum and there is real headroom above any
slice-based construction.

## Constraints
- `1 <= I, J, K <= 5`.
- `1 <= R <= 20000`.
- Deterministic exact-rational scoring; nothing is timed.

## Example
Suppose `I=J=K=2` and `S` has `B = 5` nonzero cells. A submission with `R = 2` modes that
reconstructs `S` exactly scores `min(1, 0.1*5/2) = 0.25`. The per-cell baseline
(`R = 5`) would score `0.1`. (Numbers illustrative only.)
