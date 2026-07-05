# Minimize Real Multiplications for a Small Bilinear DFT-Core Tensor

## Problem

The heart of a fast transform (FFT, short convolution, Winograd small-DFT core) is a
**bilinear map**: given two input vectors it produces an output vector as a fixed set of
bilinear forms. Such a map is a 3-D tensor `T` of shape `n1 x n2 x n3`, and the cost that
actually matters for a hardware kernel is the number of **real multiplications** — the
`R` "essential" products of two data-dependent linear combinations. Scalar pre/post
additions and multiplications by fixed constants are counted as free (they are cheap adders
and constant-scalers), exactly as in the classical Winograd multiplicative-complexity model.

You are given `T` (integer entries). Emit a **bilinear algorithm** with `R` products:

```
P_r = ( sum_i u_r[i] * a_i ) * ( sum_j v_r[j] * b_j )      r = 1..R
c_k = sum_r w_r[k] * P_r
```

that computes `T` exactly, i.e. for every `i, j, k`:

```
sum_{r=1}^{R}  u_r[i] * v_r[j] * w_r[k]  ==  T[i][j][k]
```

Minimize `R`. The tensor is generated from a **planted overcomplete decomposition**
(planted rank `> max(n1,n2,n3)`), so no polynomial-time method is known to recover the true
minimum — this is a genuinely open search with many viable strategies.

## Input (stdin)

```
n1 n2 n3
<slice k=0 : n1 rows, each n2 integers>
<slice k=1 : n1 rows, each n2 integers>
...
<slice k=n3-1>
```
Row `i` of slice `k` lists `T[i][0][k] ... T[i][n2-1][k]`.

## Output (stdout)

```
R
<product 1>
...
<product R>
```
Each product line has exactly `n1 + n2 + n3` rational tokens: first `u_r` (length `n1`),
then `v_r` (length `n2`), then `w_r` (length `n3`). Tokens are integers or fractions `p/q`
(e.g. `-1`, `3`, `2/3`). No floats in exponent form, no `nan`/`inf`.

## Feasibility

The submitted `u,v,w` must reproduce `T` **exactly** (rational arithmetic). Any mismatch,
malformed token, non-finite value, wrong token count, or `R` outside `[1, 3*n1*n2*n3+100]`
scores `0`.

## Objective

Minimize `R`, the number of bilinear multiplications.

## Scoring

Deterministic. The checker builds a trivial baseline `B` = number of `(i,j)` pairs that are
nonzero in some slice (one product per support cell — always feasible). With your feasible
count `R`:

```
Ratio = min(1.0, 0.1 * B / R)
```

Reproducing the baseline gives `0.1`; a 10x reduction caps at `1.0`. Fewer multiplications
is strictly better.

## Constraints

`2 <= n3 <= n1, n2 <= 12`. All `T[i][j][k]` are integers. Exact rational scoring only.

## Example (worked score)

Suppose `B = 48` support cells and you submit a valid bilinear algorithm with `R = 24`
products. Then `Ratio = min(1, 0.1 * 48 / 24) = 0.20`. Cutting to `R = 12` would give
`Ratio = 0.40`.
