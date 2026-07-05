# Signal-Combiner Bilinear Form: Minimize Multiplications

## Problem
A signal combiner takes two input vectors — an `m`-tap tap-vector `x` and an `n`-tap
tap-vector `y` — and must emit `P` scalar output channels, where channel `p` is the
**bilinear form**

```
out[p] = sum_{i=0..m-1} sum_{j=0..n-1}  T[p][i][j] * x[i] * y[j].
```

The coefficient tensor `T` (shape `P x m x n`, integer entries) is given. It is a
**general planted bilinear form** — it is NOT a matrix-multiplication or convolution
tensor, and its true (minimal) bilinear rank is unknown and overcomplete (it can exceed
`max(P, m, n)`), so no polynomial-time method is known to reach the optimum.

Your job: design an **arithmetic circuit** that computes all `P` outputs using as few
scalar **multiplications** as possible. Concretely, produce a *bilinear decomposition*

```
T[p][i][j] = sum_{r=1..R}  c[r][p] * a[r][i] * b[r][j]     for all p, i, j,
```

i.e. `R` rank-1 product terms. Each term `r` computes ONE multiplication
`(a[r] . x) * (b[r] . y)`; the outputs are then linear combinations of those `R`
products with weights `c[r][p]`. The cost you minimize is `R`, the multiplication count.

## Input (stdin)
```
P m n
```
followed by `P` slices; slice `p` is `m` lines of `n` integers giving `T[p][i][j]`.

## Output (stdout)
```
R
```
then `R` blocks, one per term, each block being three whitespace-separated lists:
```
a[r][0] ... a[r][m-1]        (m rational coefficients, left tap-combination)
b[r][0] ... b[r][n-1]        (n rational coefficients, right tap-combination)
c[r][0] ... c[r][P-1]        (P rational coefficients, output combiner)
```
Each number may be an integer or a rational `p/q` (e.g. `-3`, `2/5`). Non-finite tokens
(`nan`, `inf`) are rejected.

## Feasibility
The decomposition must reconstruct `T` **exactly** (exact rational arithmetic):
`sum_r c[r][p]*a[r][i]*b[r][j] == T[p][i][j]` for every `p,i,j`. Also `1 <= R <= m*n`,
and the token count must match `1 + R*(m+n+P)` exactly. Any violation scores `0`.

## Objective
Minimize `R` (the number of multiplications). Fewer is better.

## Scoring
Let `B = m*n` (the naive scheme that multiplies every `x[i]*y[j]`). A valid decomposition
of size `R` scores

```
Ratio = min(1.0, 0.1 * B / R).
```

The naive `R = m*n` scores `0.1`; a `10x` reduction caps at `1.0`. The true minimal `R`
is unknown, so the ceiling is not reachable by any known efficient method.

## Constraints
- `3 <= P <= 8`, `4 <= m = n <= 8`, integer `T`.
- Deterministic, exact-rational scoring; no time/memory is measured.

## Example (worked score, illustrative)
Suppose `m=n=4` so `B=16`. A submission that reconstructs `T` with `R=8` products scores
`min(1, 0.1*16/8) = 0.20`; one with `R=4` products scores `0.40`. A submission whose
terms fail to reconstruct even one entry scores `0.0`.
