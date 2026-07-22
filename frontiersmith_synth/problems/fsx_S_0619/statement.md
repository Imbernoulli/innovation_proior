# Cheapest Polynomial Multiplier Chip

## Problem
You are wiring a chip that multiplies two polynomials over a prime field
**F_p**. The inputs are

    a(x) = a_0 + a_1 x + ... + a_d x^d
    b(x) = b_0 + b_1 x + ... + b_d x^d

and the chip must output their full product

    c(x) = a(x) b(x) = c_0 + c_1 x + ... + c_{2d} x^{2d},   c_m = sum_{i+j=m} a_i b_j.

Scalar **multiplications** are the expensive gates; additions and multiplications
by field constants are free. So you must express the chip as a **bilinear
scheme** with as few products as possible.

A bilinear scheme with `r` products is three matrices over F_p:

* `U`  (`r x (d+1)`): each row is a left linear form `L_k(a) = sum_i U[k][i] a_i`.
* `V`  (`r x (d+1)`): each row is a right linear form `R_k(b) = sum_j V[k][j] b_j`.
* `W`  (`(2d+1) x r`): the recombination matrix.

The scheme computes product number `k` as `P_k = L_k(a) * R_k(b)` (one scalar
multiplication), then forms each output coefficient as
`c_m = sum_k W[m][k] * P_k`.

## Input (stdin)
One line: `p d`, where `p` is prime, `2 <= d <= 12`, and `3 <= d` on the graded
cases. (`p` may be smaller or larger than `2d+1`; that difference is the whole
game.)

## Output (stdout)
```
r
U[0][0] ... U[0][d]           <- r rows, d+1 ints each
   ...
V[0][0] ... V[0][d]           <- r rows, d+1 ints each
   ...
W[0][0] ... W[0][r-1]         <- (2d+1) rows, r ints each
   ...
```
All entries are integers (reduced mod `p` by the grader). `r` must satisfy
`1 <= r <= 10 (d+1)^2`.

## Feasibility
The scheme must be **exactly correct** over F_p: for every output index `m` and
every pair `(i, j)`,

    sum_k W[m][k] * U[k][i] * V[k][j]  ==  [ i + j == m ]   (mod p).

Any violation (or malformed / non-integer output) scores **0**.

## Objective
Minimize the number of products `r`.

## Scoring
Let `B = (d+1)^2` be the schoolbook product count (the checker's baseline). A
valid scheme with `r` products scores

    ratio = min(1, 0.1 * B / r).

So schoolbook scores `0.1`; halving the products roughly doubles the score; a
10x reduction caps at `1.0`. Fewer products is strictly better. The 10 graded
cases are averaged.

## Constraints
Time limit 5s, memory 512m. Each instance is tiny; the grading cost is `O((2d+1)
(d+1)^2 r)`.

## Example (worked score)
For `p = 13, d = 2` the product has 5 coefficients. Schoolbook uses
`B = 9` products and scores `0.1`. Evaluating `a` and `b` at 5 distinct field
points `0,1,2,3,4`, multiplying pointwise (5 products), and interpolating gives a
correct scheme with `r = 5`, scoring `min(1, 0.1*9/5) = 0.18`.

Note the catch: that interpolation trick needs `2d+1` **distinct** points in
F_p. When `p < 2d+1` there simply are not enough field points, and a naive
point-evaluation design stalls. The theory of arithmetic over F_p offers a way
out -- reducing modulo pairwise-coprime **irreducible polynomials** (points that
live in extension fields) lets you keep buying "cheap" coverage of the product's
degrees even over a small field. The exact minimum `r` for small `p` is not
known in closed form, so there is real room to search.
