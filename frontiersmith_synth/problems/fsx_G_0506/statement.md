# Sparse Polynomial Factor Plans for a Symbolic Compiler

## Problem
A symbolic compiler must lower several sparse polynomial expressions into a
straight-line arithmetic program. The targets are given in expanded monomial
form, but many terms hide shared sparse factors such as
`x2*x7*x11*(x3 - x5 + x9)`. Rebuilding every expanded monomial is wasteful; a
better compiler should reuse shared monomials and factor partial products when
that reduces the weighted arithmetic cost.

You are given `m` target polynomials over integer variables `x0 .. x(n-1)`.
Emit a straight-line program that computes every target exactly.

## Input (stdin)
```
n m
K_0
c d v1 v2 ... vd
...
K_{m-1}
c d v1 v2 ... vd
...
```
Each target block starts with the number of terms `K_j`. A term line represents
the monomial
```
c * x_v1 * x_v2 * ... * x_vd
```
where `c` is a nonzero integer coefficient. The generated instances use
square-free sparse monomials, but the exact checker supports repeated variables
in the submitted program.

## Output (stdout)
Print a straight-line program in this exact schema:
```
L
t0 OP A B
t1 OP A B
...
t{L-1} OP A B
OUT o0 o1 ... o{m-1}
```
`L` is the number of instruction lines. Temporaries must be defined in order as
`t0, t1, ...`; each may reference only input variables, integer constants, or
earlier temporaries.

Allowed operations are:
```
ADD A B      t = A + B
SUB A B      t = A - B
MUL A B      t = A * B
```
Each operand is one of:
- `xI` for input variable `I`;
- `tI` for an earlier temporary;
- an integer constant, either bare (`-3`) or hash-prefixed (`#-3`).

The final `OUT` line gives one operand per target polynomial.

## Feasibility
The output is rejected and scores `0` if the schema is malformed, a temporary is
out of order, an operand references a missing input or future temporary, the
program is too large, a constant/token is oversized, or any output polynomial is
not exactly equal to its target. Equivalence is checked by exact multivariate
integer-polynomial arithmetic, not by sampling or floating point.

Non-integer values such as `nan` and `inf` are invalid tokens and are rejected.

## Objective (minimize)
The weighted operation count is
```
F = 3 * (# MUL instructions) + 1 * (# ADD or SUB instructions).
```
Input variables and constants are free; all arithmetic instructions count.
Minimize `F`.

## Scoring
The checker builds an internal baseline `B` by expanding each target independently:
for every term it multiplies the variables in a chain, optionally multiplies by
the coefficient, and then accumulates the terms. The score is
```
Ratio = min(1, 0.1 * B / max(1, F)).
```
Reproducing the baseline scores about `0.1`. Full-monomial common-subexpression
reuse is useful, but the planted instances also reward partial factorization of
shared sparse cores. Finding the minimum weighted arithmetic circuit for sparse
polynomials is open-ended, so the exact optimum is not supplied.

## Constraints
- `10 <= n <= 32`
- `3 <= m <= 9`
- generated term degree is at most `7`
- submitted programs may contain at most `60000` instructions
- any intermediate polynomial with more than `250000` monomials is rejected

## Example (worked score)
For the two targets
```
y0 = x0*x1*x2 + x0*x1*x3
y1 = x0*x1*x2 - x0*x1*x3
```
a naive lowering builds the two degree-3 monomials separately for each output:
`B = 4 terms * 2 multiplications + 2 additions = 26` weighted units
because each multiplication costs `3`.

A factored program can compute
```
t0 MUL x0 x1
t1 ADD x2 x3
t2 SUB x2 x3
t3 MUL t0 t1
t4 MUL t0 t2
OUT t3 t4
```
with `F = 3*3 + 2 = 11`, so
`Ratio = min(1, 0.1 * 26 / 11) = 0.23636`. This tiny example is illustrative;
the graded cases contain more targets, more factors, and some unfactored noise.
