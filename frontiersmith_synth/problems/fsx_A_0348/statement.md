# Museum Gallery Tour — Fewest Multiplications for the Visitor Interaction Tensor

## Problem

The curator of an interactive museum models a guided gallery tour with a three-way
**interaction tensor** `T`. Entry `T[i][j][k]` is an integer weight coupling *visitor
profile* `i` (of `I`), *artwork* `j` (of `J`), and *gallery-lighting scene* `k` (of `K`).
At tour time the exhibit engine must evaluate every trilinear coupling, and each **scalar
multiplication** costs one unit of the engine's limited compute budget.

A **rank-1 term** is a triple of vectors `(a, b, c)` with `a ∈ ℚ^I`, `b ∈ ℚ^J`, `c ∈ ℚ^K`;
it contributes `a[i]·b[j]·c[k]` to every cell. A **rank-`R` decomposition** is a set of `R`
rank-1 terms whose sum reproduces `T` exactly:

```
T[i][j][k] = Σ_{r=1..R} a_r[i] · b_r[j] · c_r[k]     for all i, j, k.
```

Such a decomposition is precisely a fast bilinear scheme that evaluates the whole coupling
with `R` scalar multiplications. **Find one with `R` as small as you can.**

## Input (stdin)

```
I J K
```
followed by `I·J` lines, one per `(i, j)` pair in row-major order (`i` outer, `j` inner),
each holding the `K` integer entries `T[i][j][0 … K-1]`. Dimensions satisfy
`2 ≤ I, J, K ≤ 5`.

## Output (stdout)

```
R
<term 1>
<term 2>
...
<term R>
```
The first line is the number of rank-1 terms `R`. Each following line lists exactly
`I + J + K` rational tokens: the `I` entries of `a_r`, then the `J` entries of `b_r`, then
the `K` entries of `c_r`. A token is an integer (`-3`) or a fraction (`7/2`, `-1/4`).

## Feasibility

The decomposition is valid iff its reconstruction equals `T` **exactly** over the
rationals (no floating point, no rounding), every token is finite, and
`1 ≤ R ≤ 4·I·J·K`. Any violation scores `0`.

## Objective

Minimize `R`, the number of rank-1 terms (equivalently, scalar multiplications).

## Scoring

Let `F = R` be your term count and `B = I·J` the checker's built-in **fiber baseline**
(one term per `(i, j)` cell). The score is

```
Ratio = min(1.0, 0.1 · B / F).
```

The trivial fiber decomposition (`R = I·J`) scores `0.1`. Halving the term count doubles
the score. The planted tensor has an **overcomplete** hidden rank (strictly greater than
`max(I, J, K)`), so polynomial simultaneous-diagonalization methods cannot recover the
minimum — the true optimum is unknown and lies below every single-axis slice bound.

## Constraints

- `2 ≤ I, J, K ≤ 5`; integer tensor entries.
- Exact rational arithmetic; deterministic scoring.
- `1 ≤ R ≤ 4·I·J·K`.

## Example (worked score)

Suppose `I = J = 3`, `K = 2`, so `B = I·J = 9`.

- Emitting the 9 mode-3 fibers (`R = 9`) gives `Ratio = 0.1·9/9 = 0.100`.
- A slice-based scheme achieving `R = 6` gives `Ratio = 0.1·9/6 = 0.150`.
- A clever scheme reaching `R = 5` gives `Ratio = 0.1·9/5 = 0.180`.

(These numbers are an *illustrative* scoring calculation, not the answer to any test case.)
