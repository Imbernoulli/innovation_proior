# Ternary Rank: Build an Integer Matrix From the Fewest {-1,0,1} Outer Products

## Problem
You are given a fixed integer target matrix `M` of size `n x m`. Express it as a
sum of rank-1 outer products

```
M  =  sum_{k=1}^{r}  u_k (v_k)^T
```

where **every entry** of every factor vector `u_k` (length `n`) and `v_k` (length `m`)
comes from the **ternary alphabet `{-1, 0, 1}`**. Use as **few** terms `r` as possible.

Because each term `u_k v_k^T` is itself a rank-1 matrix with entries in `{-1,0,1}`,
the real rank of `M` is a lower bound on `r`, but the ternary alphabet is binding:
the minimum achievable `r` (the "ternary rank") is generally **strictly larger** than
the real rank and computing it exactly is NP-hard. The instance is planted as an
overcomplete ternary sum, so neither the real rank nor the planted term count is
guaranteed optimal — the true minimum is unknown.

## Input (stdin)
```
n m
row_0:  m integers
row_1:  m integers
...
row_{n-1}: m integers
```
`M[i][j]` are integers. `1 <= m < n`.

## Output (stdout)
```
r
u_0 (n integers)
v_0 (m integers)
u_1 (n integers)
v_1 (m integers)
...
u_{r-1} (n integers)
v_{r-1} (m integers)
```
That is: the term count `r`, then for each term the `n`-vector `u_k` on its own line
followed by the `m`-vector `v_k` on its own line. Whitespace is otherwise ignored.

## Feasibility
An output is feasible iff:
- `r >= 1` and the token count matches exactly `1 + r*(n+m)`;
- every entry of every `u_k` and `v_k` is in `{-1, 0, 1}` (non-integer / non-finite
  tokens are rejected);
- `sum_k u_k v_k^T == M` **exactly** (integer arithmetic).

Any violation scores `Ratio: 0.0`.

## Objective
**Minimize** `r`, the number of ternary rank-1 terms.

## Scoring
Let `B = sum_i max_j |M[i][j]|` be the checker's built-in row-wise unary baseline
(a guaranteed-feasible construction). For a feasible output with `r` terms:

```
Ratio = min(1.0, 0.1 * B / r)
```

Reproducing the baseline scores `~0.1`; using ten times fewer terms caps at `1.0`.
Infeasible or non-reconstructing outputs score `0.0`.

## Constraints
- `1 <= m < n <= 2000` (instances shipped here are small/medium).
- Entries of `M` are bounded integers.
- Deterministic, exact integer scoring only.

## Example (worked score)
Suppose `M = [[2, 0], [1, 1]]` (`n=2, m=2`, ignore the `m<n` note for this toy).
The row-wise baseline uses `B = max(2,0) + max(1,1) = 2 + 1 = 3` terms. A submission
```
r = 3
u=(1,0) v=(1,0)
u=(1,0) v=(1,0)
u=(0,1) v=(1,1)
```
reconstructs `M` exactly with `r=3`, scoring `min(1, 0.1*3/3) = 0.1`. A cleverer
2-term decomposition (if one exists) would score `min(1, 0.1*3/2) = 0.15`.
