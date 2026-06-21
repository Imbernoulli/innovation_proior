I will present the theorem known as Poincaré duality. It is one of the central statements about the topology of closed oriented manifolds, and it can be summarized as a canonical, degree-reversing isomorphism between the cohomology and the homology of such a space.

Let M be a closed n-manifold that is R-orientable, where R is a coefficient ring. The local condition that defines an n-manifold implies that near every point x the relative homology group H_n(M, M \\ {x}; R) is isomorphic to R. An R-orientation is a coherent choice of generator for each of these local groups, and when M is closed and orientable these local choices can be assembled into a single fundamental class [M] in H_n(M; R). This class is not merely a top-dimensional homology class; it is the class whose image in every local group is the chosen orientation generator.

The central operation is the cap product. For a space X there is a chain-level map that takes a p-chain and a q-cochain and returns a (p-q)-chain. The standard boundary identity for the cap product implies that it descends to a well-defined pairing on homology and cohomology, H_p(M; R) times H^q(M; R) to H_{p-q}(M; R). Feeding the fundamental class into the first slot produces the Poincaré duality map D_M : H^k(M; R) -> H_{n-k}(M; R) given by D_M(alpha) = [M] cap alpha. The theorem asserts that D_M is an isomorphism for every k. Thus a k-dimensional cohomology class, which should be thought of as a test for k-dimensional holes, is converted into an (n-k)-dimensional homology class, which should be thought of as the complementary hole itself.

Several hypotheses are doing explicit work. The dimension n of the manifold supplies the complementary degree n-k. The orientation supplies the globally coherent fundamental class; without it there is no canonical top class to cap with. Closedness means there is no boundary, so [M] lives in ordinary top homology rather than in a relative group H_n(M, boundary M; R). If M were noncompact, the natural statement would involve compactly supported cohomology. If M had a boundary, the top class would be relative and the pairing would acquire boundary terms.

The theorem has an equivalent formulation in terms of cup products. Over a field F, the universal coefficient theorem makes cohomology the linear dual of homology in the same degree. Combining this with the isomorphism D_M gives a nonsingular pairing from H^k(M; F) times H^{n-k}(M; F) to F, sending (alpha, beta) to <alpha cup beta, [M]>. The cap-cup identity relates the two viewpoints: evaluating beta on [M] cap alpha is the same, up to the standard sign convention, as evaluating alpha cup beta on [M]. This means the perfect pairing is not an extra structure; it is the same duality map read through evaluation. If a cohomology class alpha paired trivially with every complementary class, then its dual homology class [M] cap alpha would evaluate to zero against every same-degree cohomology test, and because D_M is an isomorphism alpha itself would vanish.

For smooth real manifolds the statement becomes concrete through differential forms. A closed k-form omega and a closed (n-k)-form eta wedge to a closed n-form omega wedge eta, and integration over the oriented compact manifold gives a real number. Stokes' theorem guarantees that adding an exact form to either factor does not change the integral, because the boundary term is absent. Hence there is a well-defined pairing from H^k_{dR}(M) times H^{n-k}_{dR}(M) to R, sending ([omega], [eta]) to the integral of omega wedge eta over M, and this pairing is also perfect for a closed oriented smooth manifold.

The reason the result feels like self-duality is that the manifold is not being compared with an external dual object. Its own fundamental class provides the bridge between k-dimensional tests and (n-k)-dimensional objects. This is why I refer to the theorem as Poincaré duality.

The code below gives a small computational verification of the most visible corollary for two closed oriented surfaces: a tetrahedron triangulating the 2-sphere and the seven-vertex minimal triangulation of the 2-torus. It builds the simplicial boundary operators, computes their ranks over the rationals, and checks that the Betti numbers satisfy the symmetry predicted by Poincaré duality.

```python
from fractions import Fraction

def rank_over_q(matrix):
    # Gaussian elimination over the rationals.
    if not matrix:
        return 0
    A = [row[:] for row in matrix]
    rows, cols = len(A), len(A[0])
    rank = 0
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if A[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        A[r], A[pivot] = A[pivot], A[r]
        inv = Fraction(1, A[r][c])
        for j in range(c, cols):
            A[r][j] *= inv
        for i in range(rows):
            if i != r and A[i][c] != 0:
                factor = A[i][c]
                for j in range(c, cols):
                    A[i][j] -= factor * A[r][j]
        rank += 1
        r += 1
        if r == rows:
            break
    return rank

def betti_numbers(vertices, edges, triangles):
    V, E, T = len(vertices), len(edges), len(triangles)
    edge_index = {e: i for i, e in enumerate(edges)}

    # d1: C1 -> C0, boundary of an oriented edge [i, j] is j - i.
    d1 = [[0] * E for _ in range(V)]
    for idx, (i, j) in enumerate(edges):
        d1[i][idx] = -1
        d1[j][idx] = 1

    # d2: C2 -> C1, boundary of an oriented triangle (a, b, c).
    d2 = [[0] * T for _ in range(E)]
    for t_idx, (a, b, c) in enumerate(triangles):
        terms = [(+1, b, c), (-1, a, c), (+1, a, b)]
        for sign, x, y in terms:
            if x > y:
                x, y = y, x
                sign = -sign
            d2[edge_index[(x, y)]][t_idx] += sign

    r1 = rank_over_q(d1)
    r2 = rank_over_q(d2)
    b0 = V - r1
    b1 = E - r1 - r2
    b2 = T - r2
    return [b0, b1, b2]

"""2-sphere as the boundary of a tetrahedron."""
sphere_vertices = list(range(4))
sphere_edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
sphere_triangles = [(1,2,3),(0,2,3),(0,1,3),(0,1,2)]

"""7-vertex minimal triangulation of the 2-torus."""
torus_vertices = list(range(7))
torus_edges = [(i, j) for i in range(7) for j in range(i+1, 7)]
torus_triangles = []
for i in range(7):
    torus_triangles.append((i, (i+1) % 7, (i+3) % 7))
    torus_triangles.append((i, (i+2) % 7, (i+3) % 7))

b_sphere = betti_numbers(sphere_vertices, sphere_edges, sphere_triangles)
b_torus = betti_numbers(torus_vertices, torus_edges, torus_triangles)

print("Sphere Betti numbers:", b_sphere)
print("Torus Betti numbers:", b_torus)

assert b_sphere == [1, 0, 1], "S^2 should have Betti numbers [1,0,1]"
assert b_torus == [1, 2, 1], "T^2 should have Betti numbers [1,2,1]"
for b in [b_sphere, b_torus]:
    assert b[0] == b[2], "Poincare duality predicts b0 = b2 for a closed oriented surface"
print("Poincare duality Betti symmetry verified.")
```

Running the script confirms that the sphere has Betti numbers [1, 0, 1] and the torus has [1, 2, 1], and in both cases the degree-reversing symmetry b_k = b_{2-k} holds. This is the simplest numerical shadow of the isomorphism D_M.
