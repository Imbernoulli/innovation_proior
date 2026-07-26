# Gauss-Period Bases: Load-Balancing the Multiplication Tensor

## Problem
You are given a finite field `F_{p^k}`, presented as `F_p[x] / (f(x))` for a monic
irreducible polynomial `f` of degree `k` over `F_p` (so field elements are
polynomials of degree `< k` with coefficients in `{0,...,p-1}`, multiplied modulo
`f` and reduced mod `p`). The "obvious" coordinates for this field are the
monomial (power) basis `1, x, x^2, ..., x^{k-1}` — but multiplication in these
coordinates is not necessarily cheap: reducing high-degree products modulo `f`
can spread cross terms unevenly across the output coordinates.

Your job is to **choose a different basis** `b_0, ..., b_{k-1}` of the same
`k`-dimensional `F_p`-vector space, given as an invertible `k x k` matrix `M`
over `F_p` (row `i` lists the coordinates of `b_i` in the monomial basis), so
that multiplying two elements in the new coordinates is as evenly "wired" as
possible.

Formally, the new basis defines a **structure-constant tensor** `c[i][j][l]` by
`b_i * b_j = sum_l c[i][j][l] * b_l` (ordinary field multiplication, then
re-expressed in the new basis). Think of computing a product in a circuit with
`k` parallel output **lanes**, one per coordinate `l`: lane `l` must accumulate
one term for every pair `0 <= i <= j < k` with `c[i][j][l] != 0`. The **load** of
lane `l` is the number of such pairs. Your circuit's latency is bounded by its
*busiest* lane. Choose `M` to make the busiest lane as light as possible.

Whether a basis exists that loads *every* lane **exactly equally** is a
classical number-theoretic question about `(p,k)` (Gauss periods / "optimal
normal bases"): certain `p,k` combinations admit a closed-form construction
achieving this; for the rest, no such construction is known, and load-balancing
becomes a genuine search problem. You must work out, from `p` and `k` alone,
which situation you are in.

## Input (stdin)
```
p k
f_0 f_1 ... f_{k-1}
```
`p` is prime, `2 <= k <= 20`. `f_0..f_{k-1}` are the low-to-high coefficients of
the monic degree-`k` polynomial `f(x) = x^k + f_{k-1} x^{k-1} + ... + f_0`
(guaranteed irreducible over `F_p`).

## Output (stdout)
`k*k` integers (any whitespace layout; `k` lines of `k` integers is natural),
row-major: row `i`, column `t` is the coefficient of `x^t` in `b_i`, each in
`[0, p-1]`.

## Feasibility
`M` (read as above) must be invertible modulo `p` — this is exactly the
condition for `{b_i}` to be a genuine basis. Wrong token count, an out-of-range
or non-integer entry, or a singular `M` scores `0`.

## Objective
Minimize `F = max_l w_l`, where `w_l = #{(i,j) : 0<=i<=j<k, c[i][j][l] != 0}` is
lane `l`'s load, computed from the *exact* structure-constant tensor of your
basis.

## Scoring
Let `B` be the same busiest-lane load for the monomial basis (`M = I`), computed
by the checker. With your `F`:
```
Ratio = min(1, 0.1 * B / F)
```
The monomial basis itself scores `0.1`. Halving the busiest lane's load doubles
the ratio; a tenth of `B` caps the score at `1.0`.

## Constraints
- `2 <= p <= 5`, `2 <= k <= 20`, `p^k` fits comfortably in 64-bit arithmetic
  bounds used internally; all arithmetic is exact modular arithmetic — nothing
  is timed.
- Checker runtime is `O(k^4)`, always well under the time limit.

## Example
Take `p=2, k=3, f = [1,1,0]` (`f(x) = x^3+x+1`). The monomial basis (`M=I`)
gives some busiest-lane load `B`. Suppose a candidate basis `M'` achieves
`F = B/2` (every lane's load halved). Its ratio is `min(1, 0.1*B/(B/2)) = 0.2`.
A basis achieving `F = B/10` would cap at ratio `1.0`; only bases within the
known-optimal regime get close to that.
