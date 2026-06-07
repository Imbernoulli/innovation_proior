# LLL lattice basis reduction

## Problem

Given a lattice `L` of rank `n` in `R^m` by an arbitrary integer basis
`b_1, ..., b_n`, compute, in time polynomial in `n` and the input bit-size, a new
basis of the *same* lattice that is short and nearly orthogonal — in particular
one whose first vector `b_1` is provably short relative to the shortest nonzero
lattice vector `lambda_1(L)`. Exact shortest-vector computation is hard in high
rank, so the target is a guaranteed approximation in polynomial time.

## Key idea

A lattice has no canonical short basis, but Gram-Schmidt orthogonalization of any
basis gives an orthogonal sequence `b*_1, ..., b*_n` whose lengths capture the
real geometry: the individual `|b*_i|` depend on the ordered basis, but their
product is the basis-invariant covolume `prod_i |b*_i| = d(L)`, and every nonzero
`x in L` satisfies `|x| >= |b*_i|` for the largest index `i` it reaches. So a
"good" basis is one whose Gram-Schmidt lengths decay slowly. Demand two things of
the basis, with `mu_{i,j} = <b_i, b*_j>/<b*_j, b*_j>`:

- **Size-reduction:** `|mu_{i,j}| <= 1/2` for all `j < i` (each vector is cleaned
  of integer multiples of the earlier ones; rounding `mu` to the nearest integer
  is optimal and does not change any `b*`).
- **Lovász condition** (parameter `1/4 < delta < 1`, here `delta = 3/4`):
  `|b*_i + mu_{i,i-1} b*_{i-1}|^2 >= delta |b*_{i-1}|^2`, equivalently (using
  `b*_i ⟂ b*_{i-1}`)
  `|b*_i|^2 >= (delta - mu_{i,i-1}^2) |b*_{i-1}|^2`.

When the Lovász condition fails at index `k`, swapping `b_{k-1}` and `b_k` makes
the new `b*_{k-1}` equal to the old `b*_k + mu_{k,k-1} b*_{k-1}`, whose squared
length is `< delta |b*_{k-1}|^2` — a guaranteed factor-`delta` shrink. The
potential `D = prod_{i=1}^{n-1} d_i`, where `d_i = prod_{j<=i} |b*_j|^2` is the
squared covolume of the first-`i` sublattice, changes only on swaps (size-
reduction leaves every `b*` fixed) and only by a factor `< delta`. For an
integral basis each `d_i` is a positive integer so `D >= 1`, while initially
`D <= B^{n(n-1)/2}` for `|b_i|^2 <= B`; hence there are at most `O(n^2 log B)`
swaps and the algorithm runs in polynomial time.

With `delta = 3/4`, size-reduction gives `mu_{i,i-1}^2 <= 1/4`, so the Lovász
condition forces `|b*_i|^2 >= (1/2)|b*_{i-1}|^2`. This propagates to the output
guarantees `|b_1| <= 2^{(n-1)/4} d(L)^{1/n}` and `|b_1| <= 2^{(n-1)/2} \cdot
lambda_1(L)` — the first vector is within `2^{(n-1)/2}` of the shortest vector
(and more generally `|b_j| <= 2^{(n-1)/2}` times the `j`-th successive minimum).
The general quality factor is `4/(4 delta - 1)` per index; `delta = 3/4` (giving
factor 2) balances approximation quality against the per-swap potential drop
`delta`, which controls the iteration count.

## Algorithm

1. Compute the Gram-Schmidt orthogonalization `b*_i` and the coefficients
   `mu_{i,j}`. Set `k = 2`.
2. While `k <= n`:
   - Reduce only `mu_{k,k-1}`: if `|mu_{k,k-1}| > 1/2` set
     `b_k <- b_k - round(mu_{k,k-1}) b_{k-1}` and update the Gram-Schmidt data.
     (This is all the Lovász test below depends on.)
   - If `|b*_k|^2 >= (delta - mu_{k,k-1}^2) |b*_{k-1}|^2`: finish size-reducing
     `b_k` against `b_{k-2}, ..., b_1` (for `j` from `k-2` down to `1`, if
     `|mu_{k,j}| > 1/2` set `b_k <- b_k - round(mu_{k,j}) b_j` and update), then
     set `k <- k + 1`.
   - Else swap `b_k` and `b_{k-1}`, update the Gram-Schmidt data, and set
     `k <- max(k - 1, 2)`.
3. When `k = n + 1`, the basis is reduced; return it.

## Code

```python
from fractions import Fraction


def dot(u, v):
    return sum(Fraction(a) * Fraction(b) for a, b in zip(u, v))


def gram_schmidt(B):
    """Bstar[i] = b*_i,  mu[i][j] = <b_i, b*_j> / <b*_j, b*_j>  for j < i."""
    n = len(B)
    Bstar = [None] * n
    mu = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        Bstar[i] = [Fraction(x) for x in B[i]]
        for j in range(i):
            mu[i][j] = dot(B[i], Bstar[j]) / dot(Bstar[j], Bstar[j])
            Bstar[i] = [a - mu[i][j] * c for a, c in zip(Bstar[i], Bstar[j])]
    return Bstar, mu


def lll(B, delta=Fraction(3, 4)):
    """LLL-reduce the integer rows of B (1/4 < delta < 1). Exact via Fraction."""
    B = [[Fraction(x) for x in row] for row in B]
    n = len(B)
    Bstar, mu = gram_schmidt(B)

    k = 1                                          # 0-based pointer
    while k < n:
        # reduce only mu_{k,k-1} before the test
        if abs(mu[k][k - 1]) > Fraction(1, 2):
            r = round(mu[k][k - 1])                # nearest integer
            B[k] = [a - r * b for a, b in zip(B[k], B[k - 1])]
            Bstar, mu = gram_schmidt(B)            # keep mu and b* exact

        # Lovasz condition: |b*_k|^2 >= (delta - mu_{k,k-1}^2) |b*_{k-1}|^2
        lhs = dot(Bstar[k], Bstar[k])
        rhs = (delta - mu[k][k - 1] ** 2) * dot(Bstar[k - 1], Bstar[k - 1])
        if lhs >= rhs:
            # finish size-reducing b_k against b_{k-2},...,b_0, then advance
            for j in range(k - 2, -1, -1):
                if abs(mu[k][j]) > Fraction(1, 2):
                    r = round(mu[k][j])
                    B[k] = [a - r * b for a, b in zip(B[k], B[j])]
                    Bstar, mu = gram_schmidt(B)
            k += 1                                 # advance
        else:
            B[k], B[k - 1] = B[k - 1], B[k]        # swap shrinks |b*_{k-1}|^2 < delta
            Bstar, mu = gram_schmidt(B)
            k = max(k - 1, 1)                       # retreat

    return [[int(x) for x in row] for row in B]


if __name__ == "__main__":
    basis = [[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
    for row in lll(basis):
        print(row)            # -> [0, 1, 0], [1, 0, 1], [-1, 0, 2]
```
