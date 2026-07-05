# Short-Convolution Accelerator: Fewest-Multiplier Bilinear Algorithm

## Problem
You are designing the arithmetic fabric of a **short-convolution accelerator** (the
Winograd-style kernels used inside convolutional layers). In one pass the accelerator
combines a length-`p` **signal tile** `a` with a length-`q` **filter tile** `b` and
produces `s` **pooled output taps** `c`. The wiring is a fixed *bilinear* map given by an
integer structure tensor `T` of shape `s x p x q`:

```
c[k] = sum_{i=0..p-1} sum_{j=0..q-1}  T[k][i][j] * a[i] * b[j]      (k = 0..s-1)
```

Additions and multiplications by fixed constants are cheap (routing); the expensive
resource is the **scalar multipliers** — each one multiplies two *data-dependent*
quantities. A multiplier is a term

```
m_r = ( u_r . a ) * ( v_r . b )
```

(the dot products are fixed constant combinations of the signal and filter samples), and
its result is distributed to the outputs by a constant vector `w_r`:

```
c[k] = sum_r  w_r[k] * m_r .
```

Equivalently you must write `T` as a sum of `R` rank-one bilinear terms

```
T[k][i][j] = sum_{r=1..R}  w_r[k] * u_r[i] * v_r[j] .
```

Your job: realise `T` with **as few multipliers `R` as possible** (the bilinear rank of
`T`). This is exactly the quantity Winograd's short-convolution algorithms minimise.

## Input (stdin)
```
p q s
```
then `p*q` lines. The line at position `i*q + j` (0-based, `i` outer over signal samples,
`j` inner over filter taps) lists the `s` output coefficients of the product `a[i]*b[j]`:

```
T[0][i][j]  T[1][i][j]  ...  T[s-1][i][j]
```

Dimensions satisfy `1 <= s < q < p <= 10`.

## Output (stdout)
```
R
```
followed by `R` bilinear terms. Each term is written as `p + q + s` rational numbers
(whitespace separated; you may split terms across lines freely):

```
u[0] ... u[p-1]   v[0] ... v[q-1]   w[0] ... w[s-1]
```

Each number may be an integer (`-3`), a decimal (`0.5`, `-1.25`) or an exact fraction
(`3/2`, `-7/4`). Scientific notation (`1e3`) and the tokens `nan`/`inf` are **rejected**.

## Feasibility
Let `That[k][i][j] = sum_{r=1..R} u_r[i] * v_r[j] * w_r[k]`, computed in exact rational
arithmetic. The submission is feasible iff `That == T` **exactly** at every entry. Any
parse error, wrong token count, non-finite value, `R < 1`, or reconstruction mismatch
scores 0.

## Objective
Minimize `R`, the number of scalar multipliers (bilinear rank of `T`).

## Scoring
Let `B` be the number of product fibers `a[i]*b[j]` that are actually used (nonzero for
some output tap). The naive algorithm "one multiplier per used product" always achieves
rank `R = B`. With your multiplier count `R`,

```
Ratio = min(1, 0.1 * B / R)
```

The naive per-product algorithm scores `0.1`. Halving the multiplier count doubles the
ratio; reaching a tenth of `B` caps at `1.0`. The tensor is planted with an
**overcomplete** rank (larger than every mode dimension), so spectral factorisation
cannot recover the optimum and the true minimal multiplier count lies strictly below what
simple mode-slicing achieves — genuine headroom remains.

## Constraints
- `1 <= s < q < p <= 10`; tensor entries are integers.
- `1 <= R <= 100000`.
- Deterministic exact-rational scoring; no timing, no randomness.

## Example
Suppose `p=5, q=3, s=2` and `B = 15` product fibers are used. A submission with `R = 6`
multipliers that reconstructs `T` exactly scores `min(1, 0.1*15/6) = 0.25`. The naive
per-product algorithm (`R = 15`) scores `0.1`.
