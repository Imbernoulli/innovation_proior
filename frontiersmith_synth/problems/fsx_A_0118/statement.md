# Signal-Grid Actuation Tensor: Fewest Shared Multiplier Units

## Problem
A city intersection controller stores a three-dimensional **actuation-response
table** `T[i][j][k]` of integer gain coefficients over

- `i` = approach direction (`a` of them),
- `j` = movement type (`b` of them, e.g. through / left / right / bus-priority),
- `k` = signal phase (`c` of them).

The naive controller computes the responses with **one hardware multiply per
nonzero coefficient**. You must instead implement the exact same table with a
small number `R` of **shared multiplier units**. Each unit `r` is a *separable
gain* described by three weight vectors `u_r` (length `a`, over approaches),
`v_r` (length `b`, over movements) and `w_r` (length `c`, over phases), and it
contributes `u_r[i] * v_r[j] * w_r[k]` to entry `(i, j, k)`.

The controller realizes

```
T[i][j][k] = sum_{r=1..R} u_r[i] * v_r[j] * w_r[k]      for all i, j, k
```

i.e. a **rank-`R` CP decomposition** of the tensor `T`. Fewer units `R` means
fewer physical multipliers — minimize `R`.

## Input (stdin)
```
a b c
```
followed by the tensor body: `a` blocks, each block has `b` lines, and each line
lists the `c` integer phase coefficients `T[i][j][0..c-1]`. (Equivalently: after
the header, `a*b*c` integers in the order `i` outer, then `j`, then `k` inner.)
All dimensions satisfy `1 <= a <= b < c <= 5`.

## Output (stdout)
```
R
<stage 1: a+b+c rationals>
<stage 2: a+b+c rationals>
...
<stage R: a+b+c rationals>
```
Each stage line lists `u_r[0..a-1]`, then `v_r[0..b-1]`, then `w_r[0..c-1]`
(total `a+b+c` numbers). Each number may be an integer, a fraction `p/q`, or a
plain decimal such as `-0.5`. Non-finite tokens (`nan`, `inf`) and scientific
notation are rejected.

## Feasibility
The submitted stages must reproduce `T` **exactly** under rational arithmetic:
`sum_r u_r[i] v_r[j] w_r[k] == T[i][j][k]` for every `(i,j,k)`. Any mismatch,
malformed stage, wrong token count, or out-of-range factor scores `0`.

## Objective
Minimize `R`, the number of shared multiplier units (scalar-multiplication
count of the separable implementation).

## Scoring
Let `B` = the number of nonzero entries of `T` (the naive one-multiply-per-entry
implementation, always feasible). With `R` your feasible unit count,

```
Ratio = min(1, 0.1 * B / R)
```

Reproducing the naive table (`R = B`) scores `0.1`; a 10x reduction caps at
`1.0`. The tensor is **planted with an overcomplete rank** (more planted stages
than the largest dimension), so pencil / Jennrich-style methods cannot recover
the optimum and the true minimal `R` is genuinely unknown.

## Constraints
- `1 <= a <= b < c <= 5`; integer coefficients.
- At most `5000` stages; `|numerator|, |denominator| <= 1e9` per factor.
- Deterministic scoring: exact rational arithmetic, no timing.

## Example (worked score)
Suppose `a=2, b=3, c=4` and `T` has `B = 24` nonzero entries. The naive table
uses `R = 24` units -> `Ratio = 0.1*24/24 = 0.1`. Slicing along the phase axis
and rank-factorizing each of the 4 frontal `2x3` slices (rank 2 each) gives
`R = 8` -> `Ratio = 0.1*24/8 = 0.3`. Choosing instead the best of the three axes
gives `R = 6` -> `Ratio = 0.1*24/6 = 0.4`. Any exact decomposition with still
fewer units scores higher, up to the cap.
