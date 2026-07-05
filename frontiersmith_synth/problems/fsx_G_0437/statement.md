# Tensor Lab: Minimal-Multiplication Bilinear Decomposition

## Problem

In the tensor lab you are handed a 3-dimensional array `T` of shape `(a, b, c)`
with integer entries. A **bilinear decomposition** (CP decomposition) writes `T`
as a sum of `R` rank-1 terms:

```
T[i][j][k]  =  sum_{r=1..R}  u_r[i] * v_r[j] * w_r[k]      for all i, j, k
```

Each term `u_r (x) v_r (x) w_r` costs exactly **one scalar multiplication** when
the bilinear form is evaluated, so `R` is the multiplication count of the
resulting straight-line bilinear algorithm. Your job is to reconstruct `T`
**exactly** using **as few terms `R` as possible**.

The tensor is generated so that every mode-`c` slice is a rank-2 matrix, while
the overall tensor rank is **overcomplete** (larger than `max(a, b, c)`). The
true minimal `R` is therefore genuinely unknown — simultaneous-diagonalization
methods (which require rank at most the largest dimension) do not apply, so there
is real room to be clever.

## Input (stdin)

```
a b c
```
followed by the `c` slices of `T`, printed one after another. Slice `k`
(`k = 0..c-1`) is printed as `a` lines of `b` integers, where line `i` holds
`T[i][0][k] T[i][1][k] ... T[i][b-1][k]`.

All entries are integers.

## Output (stdout)

```
R
<term 1: u (a numbers)>
<term 1: v (b numbers)>
<term 1: w (c numbers)>
<term 2: u ...>
...
```

The first token is the number of terms `R` (an integer `>= 1`). Then, for each
term in order, print three lines: the `a`-vector `u_r`, the `b`-vector `v_r`, and
the `c`-vector `w_r`. Numbers may be integers or exact rationals written as
`p/q` (e.g. `-3/2`). Non-finite tokens (`nan`, `inf`) and scientific notation are
rejected. Whitespace between tokens is free-form.

## Feasibility

The decomposition is feasible iff it reconstructs `T` **exactly** over the
rationals:
`sum_r u_r[i] v_r[j] w_r[k] == T[i][j][k]` for every `(i, j, k)`. Any mismatch,
malformed schema, wrong token count, `R < 1`, oversized `R`, or non-finite value
scores `0`.

## Objective (minimize)

Minimize `R`, the number of rank-1 terms (scalar multiplications).

## Scoring

The checker builds its own baseline `B` = the trivial mode-`c` flattening
(one term per nonzero `(i,j)` fibre, i.e. `B = a*b` nonzero fibres). With
`F = R` your term count,

```
Ratio = min(1.0, 0.1 * B / F)
```

Reproducing the baseline gives `Ratio = 0.1`; a 10x reduction caps at `1.0`.
Infeasible outputs score `0`.

## Constraints

- `5 <= a = b <= 14`, `4 <= c <= 13` (difficulty ladder over test ids).
- Entries are small integers; use exact integer/rational arithmetic.

## Example (worked score)

Suppose `a=b=5, c=4` and the trivial baseline has `B = 25` nonzero fibres.
- Emitting the 25 trivial fibre terms gives `R = 25`, `Ratio = 0.1*25/25 = 0.1`.
- Slicing along mode `c` and rank-2-factoring each of the 4 slices gives
  `R = 2*4 = 8`, `Ratio = 0.1*25/8 = 0.3125`.
- A decomposition that exploited cross-slice structure to reach, say, `R = 6`
  would score `Ratio = 0.1*25/6 = 0.4167`.

(Illustrative numbers only; your instance differs.)
