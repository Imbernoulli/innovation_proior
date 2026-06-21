I want to explain the canonical counting identity for meromorphic functions on an algebraic curve, which I will call the Riemann–Roch theorem for nonsingular projective curves. The problem begins with a very concrete question. Suppose I am given a compact Riemann surface or, equivalently, a nonsingular projective curve over an algebraically closed field. I pick a finite set of points and declare how large a pole a meromorphic function may have at each of those points, or equivalently I require zeros of certain multiplicities. How many linearly independent functions satisfy these local conditions?

At first the answer looks almost local. If I allow poles of total order m, then naively I expect m independent principal-part coefficients, plus the constant function that has no pole anywhere. So a natural first guess is that the space of functions has dimension m plus one. This is exactly right when the curve is a sphere, because every local principal part can be patched together globally without obstruction. The sphere has no holes, no cycles, and no hidden consistency conditions.

Once the curve has positive genus, however, the naive count starts to fail. The surface now has nontrivial one-cycles, and a candidate expression built from local principal parts is generally multivalued. To make it single-valued I have to impose period conditions, one for each independent holomorphic differential. Since there are g independent holomorphic differentials on a curve of genus g, the next natural correction subtracts g from the naive count. This leads to the provisional formula m plus one minus g, which is the generic answer when the period equations are independent.

But independence is not automatic. In special configurations the period conditions can become dependent, and when they do the provisional count is too small. The dependencies are measured by dual objects: differentials that vanish to sufficiently high order at the prescribed pole points. Each independent such differential gives a linear relation among the period equations and therefore frees one extra dimension in the space of functions. If the dimension of this dual vanishing space is q, then the true count becomes m plus one minus g plus q.

This is the content of the Riemann–Roch theorem. To state it cleanly I introduce a divisor D, which is the formal bookkeeping device that records the allowed poles and required zeros point by point. The space L(D) consists of all meromorphic functions f whose divisor plus D is effective, together with zero, and l(D) denotes its dimension. The canonical divisor K captures the holomorphic differentials on the curve. The theorem then says that l(D) minus l(K minus D) equals the degree of D plus one minus g.

The left-hand side is what I can actually compute in practice: the dimension I want minus the dimension of the dual correction space. The right-hand side is the naive local budget corrected by the genus. The theorem is remarkable because it converts a delicate existence question into an exact identity. The term l(K minus D) is precisely the q I described above; it counts the independent differentials that impose relations among the period conditions.

When D is sufficiently large, specifically when its degree exceeds 2g minus 2, the divisor K minus D has negative degree. A space of functions attached to a divisor of negative degree can contain only the zero function, because a nonzero meromorphic function must have degree zero and therefore cannot have more zeros than poles. Hence l(K minus D) vanishes, and the Riemann–Roch formula collapses to l(D) equals deg(D) plus one minus g. This is the generic case, and it matches the provisional count I started with.

In the small or special cases, the correction term is nonzero and the theorem explains exactly where the extra functions come from. For example, on an elliptic curve where g equals one, the canonical divisor is principal and has degree zero. The formula then reduces to l(D) equals deg(D) whenever D has positive degree, which is the classical statement that every degree-d effective divisor on an elliptic curve gives rise to a d-dimensional space of functions. For lower genus or for divisors supported at special points, the correction term captures the geometry that the naive count misses.

The theorem is usually proved in two stages. The first stage establishes Riemann's inequality, which shows that l(D) is at least deg(D) plus one minus g. This can be done by constructing enough functions with controlled poles, for instance by using the projective embedding of the curve or by explicit Mittag-Leffler-type sums on the Riemann surface. The second stage, contributed by Roch, identifies the difference between the true dimension and Riemann's lower bound with l(K minus D). This duality argument is what turns an inequality into an equality and is the reason the canonical divisor appears.

One intuitive way to see the duality is through Serre duality or, more classically, through the residue pairing. A meromorphic differential with appropriate vanishing at the pole points can be paired with principal parts of functions via residues. When this pairing is degenerate, it signals a relation among the period constraints, and the dimension of the degeneracy is exactly l(K minus D). Thus the Riemann–Roch formula is not merely a numerical coincidence but a rank statement about a linear map from local data to global constraints.

The theorem extends naturally to higher-rank vector bundles via the Hirzebruch–Riemann–Roch theorem, and it underlies many constructions in algebraic geometry, from the classification of curves to the definition of the Jacobian. In machine learning and coding theory, analogous Euler-characteristic formulas appear whenever one counts global sections of sheaves with local constraints. But the original curve version remains the cleanest illustration of the principle.

To make the identity concrete, I will verify it on a family of hyperelliptic curves. Consider a curve given by y squared equals f(x), where f is a polynomial of odd degree 2g plus one and the characteristic is not two. Such a curve has genus g, and there is a single point at infinity. The function x has pole order two at infinity, while y has pole order 2g plus one. A basis for the space L(n times infinity) is therefore easy to write down explicitly: it consists of the monomials x to the i for 2i at most n, together with the products x to the j times y for 2j plus 2g plus one at most n. By enumerating this basis and comparing with the Riemann–Roch prediction, the equality can be checked for many values of n and g without any advanced machinery.

```python
def l_infinity(n, g):
    """Dimension of L(n * infinity) on the hyperelliptic curve
    y^2 = f(x) with deg(f) = 2*g + 1 (odd), char != 2.
    The unique point at infinity is one place; x has pole order 2
    and y has pole order 2*g + 1 there.
    """
    if n < 0:
        return 0
    # basis {x^i : 2*i <= n}
    from_x = n // 2 + 1
    # basis {x^j * y : 2*j + (2*g + 1) <= n}
    rem = n - (2 * g + 1)
    from_xy = rem // 2 + 1 if rem >= 0 else 0
    return from_x + from_xy


def verify_riemann_roch(g, n_max=12):
    """Check l(n*infty) - l(K - n*infty) = n + 1 - g,
    where K ~ (2*g - 2) * infty on this hyperelliptic model.
    """
    K_degree = 2 * g - 2
    print(f"Hyperelliptic curve genus g = {g}, canonical degree = {K_degree}")
    for n in range(-2, n_max + 1):
        lD = l_infinity(n, g)
        # K - n*infty has degree K_degree - n.
        if K_degree - n < 0:
            lKminusD = 0
        else:
            lKminusD = l_infinity(K_degree - n, g)
        lhs = lD - lKminusD
        rhs = n + 1 - g
        status = "OK" if lhs == rhs else "FAIL"
        print(f"  n={n:3d}: l(D)={lD}, l(K-D)={lKminusD}, "
              f"l(D)-l(K-D)={lhs:3d}, deg(D)+1-g={rhs:3d} [{status}]")


if __name__ == "__main__":
    for genus in (1, 2, 3):
        verify_riemann_roch(genus)
        print()
```

This small program computes the dimensions directly from the explicit basis and confirms that the Riemann–Roch identity holds for every n in the tested range. It also illustrates the two regimes: for large n the correction term vanishes and the dimension grows linearly with slope one, while for small n the correction term adjusts the count to keep the Euler characteristic exact. In summary, the Riemann–Roch theorem is the statement that local pole freedom, genus-imposed global constraints, and the dual obstruction space fit together in the single identity l(D) minus l(K minus D) equals deg(D) plus one minus g.
