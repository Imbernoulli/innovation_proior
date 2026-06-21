Consider a compact smooth manifold X and an elliptic differential or pseudodifferential operator P acting between complex vector bundles over X. Elliptic regularity tells us that, once the spaces of smooth sections are completed in suitable Sobolev norms, P becomes a Fredholm operator. Its analytic index

ind(P) = dim ker(P) - dim coker(P)

is therefore a well-defined integer. The difficulty is that this integer is built from the solutions of a partial differential equation, and one would like to compute it without ever solving the equation. Direct spectral analysis is not enough: the kernel and cokernel can change drastically under compact or lower-order perturbations, even though their difference stays constant. Classical formulas such as the Euler characteristic for the de Rham complex or Hirzebruch-Riemann-Roch for the Dolbeault complex give the right kind of answer in special cases, but they do not cover arbitrary elliptic operators on arbitrary compact smooth manifolds. Heat-kernel methods are powerful, yet for a general elliptic operator the small-time asymptotics involve too many derivatives of the total symbol; they become truly clean only for Dirac-type operators where Clifford multiplication forces the cancellations that produce characteristic forms. What is missing is a single topological invariant that captures the Fredholm index using only the principal symbol and the topology of X.

The Atiyah-Singer Index Theorem supplies exactly that invariant. The theorem says that the analytic Fredholm index of P equals the topological index of the principal symbol of P. In more detail, the principal symbol of P is a bundle isomorphism over the cotangent bundle away from the zero section, so it defines a compactly supported K-theory class [sigma(P)] in K^0_c(T^*X). The analytic index is the homomorphism that sends this class to dim ker(P) - dim coker(P). The topological index is the K-theoretic pushforward of [sigma(P)] to a point, which lands in K^0(pt) = Z. The theorem asserts that these two homomorphisms are equal:

a-ind([sigma(P)]) = t-ind([sigma(P)]).

Passing from K-theory to cohomology via the Chern character and the Thom isomorphism gives the more familiar integral formula

ind(P) = < Todd(TX tensor C) cup phi^{-1}(ch [sigma(P)]), [X] >.

The placement of the Thom isomorphism depends on orientation conventions, but the invariant content is fixed: the analytic defect of the elliptic PDE is a topological invariant of the symbol bundle over the cotangent bundle and the manifold that carries it.

This is the right formulation because every structural property of the Fredholm index is mirrored by a corresponding construction in K-theory. Lower-order terms are compact perturbations, so the analytic index factors through the principal symbol class. The symbol itself is exactly the data K-theory was invented to study: two vector bundles glued by an isomorphism outside a compact set. The pushforward to a point is forced by functoriality under diffeomorphisms, additivity under direct sums, homotopy invariance, excision, compatibility with Thom products, and normalization on the model class over a point. Once both the analytic and topological index maps are shown to satisfy these properties, uniqueness forces them to be the same. Special cases such as Gauss-Bonnet-Chern for the de Rham complex, Hirzebruch-Riemann-Roch for Dolbeault operators, and the signature theorem for the signature operator all emerge as particular computations of the same universal map.

The code below implements the cohomological side of the theorem for a few concrete cases. It works with truncated cohomology rings, computes Chern characters and Todd classes, and evaluates the Atiyah-Singer integral against the fundamental class. The examples verify the theorem for the Euler characteristic of the 2-sphere and for the Hirzebruch-Riemann-Roch formula on the complex projective line.

```python
"""
A small symbolic verification of the Atiyah-Singer Index Theorem
on two model manifolds: the real 2-sphere and the complex
projective line CP^1.
"""

from fractions import Fraction
from typing import List


class CohomologyRing:
    """Truncated polynomial ring in one variable x, with x^(dim+1)=0."""

    def __init__(self, dim: int):
        self.dim = dim
        self.c: List[Fraction] = [Fraction(0)] * (dim + 1)

    @classmethod
    def from_constant(cls, dim: int, value: Fraction):
        r = cls(dim)
        r.c[0] = value
        return r

    def __add__(self, other):
        assert self.dim == other.dim
        r = CohomologyRing(self.dim)
        r.c = [a + b for a, b in zip(self.c, other.c)]
        return r

    def __mul__(self, other):
        if isinstance(other, CohomologyRing):
            assert self.dim == other.dim
            r = CohomologyRing(self.dim)
            for i, a in enumerate(self.c):
                for j, b in enumerate(other.c):
                    if i + j <= self.dim:
                        r.c[i + j] += a * b
            return r
        # scalar multiplication on the right
        r = CohomologyRing(self.dim)
        r.c = [other * a for a in self.c]
        return r

    def __rmul__(self, scalar: Fraction):
        r = CohomologyRing(self.dim)
        r.c = [scalar * a for a in self.c]
        return r

    def integral(self) -> Fraction:
        """Pairing with the fundamental class, i.e. coefficient of x^dim."""
        return self.c[self.dim]


def exp_class(a: CohomologyRing) -> CohomologyRing:
    """Exponential in the truncated ring: exp(a) = sum a^k/k!."""
    result = CohomologyRing.from_constant(a.dim, Fraction(1))
    term = CohomologyRing.from_constant(a.dim, Fraction(1))
    for k in range(1, a.dim + 1):
        term = term * a * Fraction(1, k)
        result = result + term
    return result


def todd_class(c1: CohomologyRing, c2: CohomologyRing = None) -> CohomologyRing:
    """
    Todd class up to degree 4:
        td = 1 + c1/2 + (c1^2 + c2)/12 + c1*c2/24 + ...
    For surfaces we only need the terms shown.
    """
    one = CohomologyRing.from_constant(c1.dim, Fraction(1))
    td = one + c1 * Fraction(1, 2)
    if c2 is not None:
        td = td + (c1 * c1 + c2) * Fraction(1, 12)
        td = td + (c1 * c2) * Fraction(1, 24)
    return td


def riemann_roch_index(bundle_ch: CohomologyRing,
                       tangent_c1: CohomologyRing) -> Fraction:
    """Atiyah-Singer for a Dolbeault-type operator twisted by E:
       chi(E) = integral_X td(TX) ch(E).
    """
    td = todd_class(tangent_c1)
    return (td * bundle_ch).integral()


def euler_index(euler_class: CohomologyRing) -> Fraction:
    """Atiyah-Singer for the de Rham complex:
       chi(X) = integral_X e(TX).
    """
    return euler_class.integral()


# ---------------------------------------------------------------------------
# Example 1: the 2-sphere, real dimension 2.
# H^*(S^2) = Z[u]/(u^2), u integrated over S^2 gives 1.
# The Euler class of TS^2 is 2u, so chi(S^2) = 2.
# ---------------------------------------------------------------------------
S2 = CohomologyRing(1)
u = CohomologyRing(1)
u.c[1] = Fraction(1)
euler_S2 = u * Fraction(2)
print("S^2 analytic index (Euler characteristic) =", euler_index(euler_S2))

# ---------------------------------------------------------------------------
# Example 2: CP^1, complex dimension 1, real dimension 2.
# H^*(CP^1) = Z[H]/(H^2), integral of H is 1.
# c1(T CP^1) = 2H, so td(T CP^1) = 1 + H.
# For the holomorphic line bundle O(n), ch(O(n)) = exp(nH) = 1 + nH.
# Hirzebruch-Riemann-Roch gives chi(O(n)) = n + 1.
# ---------------------------------------------------------------------------
CP1 = CohomologyRing(1)
H = CohomologyRing(1)
H.c[1] = Fraction(1)

c1_tangent_CP1 = H * Fraction(2)

for n in (0, 1, 2, 3):
    ch_O_n = exp_class(H * Fraction(n))
    idx = riemann_roch_index(ch_O_n, c1_tangent_CP1)
    print(f"CP^1 chi(O({n})) via Atiyah-Singer = {idx}  (expected {n + 1})")

# ---------------------------------------------------------------------------
# A generic check: the index map is additive under direct sum of symbols.
# If we take two line bundles O(m) and O(n), their direct sum should have
# index (m+1) + (n+1).
# ---------------------------------------------------------------------------
m, n = 2, 4
ch_sum = exp_class(H * Fraction(m)) + exp_class(H * Fraction(n))
idx_sum = riemann_roch_index(ch_sum, c1_tangent_CP1)
print(f"CP^1 additivity check: chi(O({m}) + O({n})) = {idx_sum}  "
      f"(expected {(m + 1) + (n + 1)})")
```
