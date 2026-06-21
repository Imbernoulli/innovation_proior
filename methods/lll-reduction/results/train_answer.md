I propose the canonical name Lenstra–Lenstra–Lovász lattice basis reduction, or simply LLL reduction, for the problem of turning an arbitrary integer basis of a lattice into a basis whose vectors are short and nearly orthogonal while keeping every step inside the same lattice. The input is a lattice L of rank n in R^m, presented as n linearly independent integer vectors b_1, ..., b_n. The legal moves are exactly the unimodular ones: add an integer multiple of one basis vector to another, swap two basis vectors, or negate a vector. Each such move is represented by an integer matrix of determinant plus or minus one, so the lattice itself never changes. The goal is to reach, in time polynomial in n and in the bit-size of the input, a basis whose first vector is provably short relative to the true shortest nonzero vector of L.

The difficulty is that there is no canonical short basis. The same lattice has infinitely many bases, related by GL_n(Z), and ordinary Euclidean lengths of the basis vectors are not invariant under those moves. What is invariant is the covolume d(L), the square root of the determinant of the Gram matrix of inner products. So any useful notion of quality must be one the unimodular group cannot inflate away. The right invariant quantities come from Gram-Schmidt orthogonalization. From the given ordered basis we build an orthogonal sequence b*_1, ..., b*_n by projecting each b_i orthogonally onto the orthogonal complement of the span of the earlier vectors. The coefficients are mu_{i,j} = <b_i, b*_j> / <b*_j, b*_j> for j < i. Because the Gram-Schmidt vectors are pairwise orthogonal, the product of their lengths equals the covolume: prod_i |b*_i| = d(L). Moreover, any nonzero lattice vector x = sum r_i b_i, with integers r_i, can be rewritten as x = sum r'_i b*_i, and the change of basis is upper triangular with ones on the diagonal. Hence for the largest index i with r_i nonzero we have r'_i = r_i, a nonzero integer, and therefore |x| >= |b*_i|. So the Gram-Schmidt lengths are the real currency of the geometry: they lower-bound the lengths of lattice vectors that reach each index, and their product is fixed by the lattice alone.

A good basis, then, is one whose Gram-Schmidt lengths decay slowly. If |b*_i| stays comparable to |b*_{i-1}| all along, then every nonzero lattice vector is at least roughly |b*_1|, and b_1, whose length starts at |b*_1|, is essentially as short as anything in L. Two conditions capture this. The first is size-reduction: for every j < i, the Gram-Schmidt coefficient mu_{i,j} should satisfy |mu_{i,j}| <= 1/2. This says that b_i has been cleaned of integer multiples of earlier basis vectors as far as possible; the best single move is b_i <- b_i - round(mu_{i,j}) b_j, which lands the coefficient in the interval [-1/2, 1/2]. Size-reduction does not change any b*_i, so it is free in the Gram-Schmidt-length currency. The second condition is the Lovasz condition with a parameter delta in (1/4, 1): |b*_i + mu_{i,i-1} b*_{i-1}|^2 >= delta |b*_{i-1}|^2. Because b*_i is orthogonal to b*_{i-1}, this is equivalent to |b*_i|^2 >= (delta - mu_{i,i-1}^2) |b*_{i-1}|^2. It says that the projected rank-two pair formed by b_{i-1} and b_i is already reduced enough in the sense of Gauss-Lagrange rank-two reduction; no adjacent swap would improve the leading Gram-Schmidt length by more than a factor of delta.

The constant delta is the crucial relaxation that makes polynomial time possible. If we required every swap to give any improvement at all, a swap could shrink the leading length by a factor arbitrarily close to one, and we would have no bound on the number of swaps. By insisting on a definite factor delta < 1, each useful swap multiplies a global potential by less than delta. Define d_i = det(<b_j, b_l>)_{1 <= j,l <= i} = prod_{j <= i} |b*_j|^2, the squared covolume of the rank-i sublattice spanned by the first i vectors, and define D = prod_{i=1}^{n-1} d_i. Size-reduction leaves every b*_i unchanged, so it leaves D unchanged. A swap of b_{k-1} and b_k only affects d_{k-1}: the pair's covolume is invariant, and all d_i with i different from k-1 are untouched. When the Lovasz condition fails, the new d_{k-1} is less than delta times the old one, so D drops by a factor below delta. For an integral basis each d_i is a positive integer, hence D >= 1, while initially |b_i|^2 <= B gives D <= B^{n(n-1)/2}. Therefore the number of swaps is at most O(n^2 log B), and the algorithm terminates in polynomial time. Over the reals the same argument works using Minkowski's convex-body bound to give a positive lower bound on D, but the integer floor is the clean polynomial guarantee.

The standard choice is delta = 3/4. Then, after size-reduction, mu_{i,i-1}^2 <= 1/4, and the Lovasz condition implies |b*_i|^2 >= (1/2) |b*_{i-1}|^2. From this slow decay one obtains the classical quality guarantees: the product of the raw basis lengths satisfies prod_i |b_i| <= 2^{n(n-1)/4} d(L), and the first vector satisfies |b_1| <= 2^{(n-1)/4} d(L)^{1/n}. More importantly, relative to the true shortest vector lambda_1(L), we get |b_1| <= 2^{(n-1)/2} lambda_1(L). The same argument bounds each b_j by 2^{(n-1)/2} times the j-th successive minimum, so the whole reduced basis is short, not merely its first vector. The general quality factor per index is 4/(4 delta - 1); delta = 3/4 gives the clean base-two bounds above. Pushing delta toward one improves the approximation factor but weakens the per-swap potential drop, so the polynomial-time bound blows up; choosing delta = 3/4 balances the two.

The algorithm is implemented as a single sweep with a pointer k, starting at k = 2 in one-based indexing (k = 1 in zero-based code). The invariant is that the prefix before k is already fully size-reduced and satisfies the Lovasz condition at every internal step. At each k, we first size-reduce b_k only against its immediate predecessor b_{k-1}; this is the only coefficient the upcoming test depends on. Then we test the Lovasz condition. If it holds, we finish size-reducing b_k against b_{k-2}, ..., b_1 and advance k by one. If it fails, we swap b_{k-1} and b_k, which shrinks the potential D by the required factor, and retreat k by one to recheck the previous pair, since the swap may have disturbed it. When k reaches n + 1 the entire basis is reduced and we return it. For exactness the arithmetic can be carried out over the rationals with Fraction objects; all denominators stay controlled, and the algorithm remains polynomial. A self-contained Python implementation is given below.

```python
from fractions import Fraction


def dot(u, v):
    return sum(Fraction(a) * Fraction(b) for a, b in zip(u, v))


def gram_schmidt(B):
    """Return orthogonalized Bstar and coefficients mu with Bstar[i] = b*_i."""
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
    """LLL-reduce the integer rows of B (1/4 < delta < 1) using exact arithmetic."""
    B = [[Fraction(x) for x in row] for row in B]
    n = len(B)
    Bstar, mu = gram_schmidt(B)

    k = 1  # zero-based pointer; corresponds to b_{k+1} in one-based notation
    while k < n:
        # Reduce only mu_{k,k-1} before the Lovasz test.
        if abs(mu[k][k - 1]) > Fraction(1, 2):
            r = round(mu[k][k - 1])
            B[k] = [a - r * b for a, b in zip(B[k], B[k - 1])]
            Bstar, mu = gram_schmidt(B)

        # Lovasz condition: |b*_k|^2 >= (delta - mu_{k,k-1}^2) |b*_{k-1}|^2.
        lhs = dot(Bstar[k], Bstar[k])
        rhs = (delta - mu[k][k - 1] ** 2) * dot(Bstar[k - 1], Bstar[k - 1])
        if lhs >= rhs:
            # Finish size-reducing b_k against the rest of the prefix, then advance.
            for j in range(k - 2, -1, -1):
                if abs(mu[k][j]) > Fraction(1, 2):
                    r = round(mu[k][j])
                    B[k] = [a - r * b for a, b in zip(B[k], B[j])]
                    Bstar, mu = gram_schmidt(B)
            k += 1
        else:
            # Swap shrinks |b*_{k-1}|^2 by a factor below delta; retreat to recheck.
            B[k], B[k - 1] = B[k - 1], B[k]
            Bstar, mu = gram_schmidt(B)
            k = max(k - 1, 1)

    return [[int(x) for x in row] for row in B]


if __name__ == "__main__":
    basis = [[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
    reduced = lll(basis)
    print("Reduced basis:")
    for row in reduced:
        print(row)
```
