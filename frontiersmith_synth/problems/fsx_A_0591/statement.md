# Reshape a Fixed Polynomial: Fewest Online Multiplications

## Problem
You are given the coefficients of a fixed univariate polynomial
`p(x) = a_0 + a_1 x + ... + a_d x^d` (integer coefficients, `a_d != 0`).
Emit a **straight-line arithmetic program** that computes `p(x)` for the symbolic
input `x`. The coefficients are known *offline*, so any arithmetic on constants
you derive from them is **free**; only *online* multiplications between two
`x`-dependent quantities are charged. Your goal is to minimise that charge.

## Machine model (output = your program)
Register `r0` is the input `x`. Each line appends a new register. Instructions:

```
CON  c        r = c                (free)   c = integer or rational p/q
SMUL i c      r = c * r_i          (free)   scalar (constant) multiply
SADD i c      r = r_i + c          (free)
ADD  i j      r = r_i + r_j        (free)
SUB  i j      r = r_i - r_j        (free)
MUL  i j      r = r_i * r_j        (COST 1) nonscalar multiply
RET  i        output := r_i        (exactly once, as the last line)
```

`i`,`j` are indices of already-defined registers. Only `MUL` (a product of two
registers) costs; every constant load, scalar multiply, and addition is free
because the constants are precomputed from the fixed coefficients. Let `F` be the
number of `MUL` lines you use.

## Feasibility
Your program must be a **formal identity**: evaluated as a polynomial in the
indeterminate `x`, the returned register must equal `p(x)` exactly (checked with
exact rational arithmetic, coefficient by coefficient). Any malformed line, index
out of range, unparsable/`nan`/`inf` constant, missing/duplicate `RET`, an
intermediate register of degree above `2d`, or a value that is not identically
`p(x)` makes the submission infeasible (score 0).

## Objective
Minimise `F`, the online multiplication count.

## Scoring
Let `B_hi = d(d-1)/2` be the multiplications used by the naive method
(compute every power `x^2..x^d` from scratch, scale by its coefficient, add), and
let `B_lo` be a strong baby-step / giant-step reference count for degree `d`.
Your score is a fixed, monotone-decreasing function of `F`:

```
score = clip( 0.1 + 0.7 * (ln B_hi - ln F) / (ln B_hi - ln B_lo),  0, 1 )
```

So `F = B_hi` scores `0.1`, matching `B_lo` scores `0.8`, and beating the
reference climbs toward `1.0`. Horner's rule (`d` multiplies) sits in between:
it *feels* optimal, but it is not, because you may spend unlimited free offline
arithmetic to reshape `p` and reuse a small set of nonscalar products.

## The insight
Horner charges one online multiply per degree: `d` total, and it is provably
optimal when the coefficients are unknown. Here they are **known**. Split `p`
into blocks of width `k ≈ sqrt(d)`: precompute `x^2..x^k` (about `k` products),
form each block `B_j(x)=Σ a_{jk+t} x^t` with only free scalar multiplies and adds,
then combine the blocks by a Horner recurrence in `y = x^k` (about `d/k` products).
Total ≈ `2*sqrt(d)` online multiplies — far below `d`. The free offline work
(the block coefficients, `y = x^k`) is what buys the reduction.

## Constraints
`4 <= d <= 30`. Coefficients fit in a machine word. Time limit 5 s, memory 512 MB.
Program at most 20000 lines.

## Example (scoring only, not the intended solution)
For `d = 12`, Horner uses `F = 12` and the naive method `B_hi = 66`; a
baby-step/giant-step layout reaches `F = 6`. With `B_lo = 6`, Horner scores about
`0.60` while the `F = 6` program scores `0.80`. Finding a program with `F < 6`
would score even higher.
