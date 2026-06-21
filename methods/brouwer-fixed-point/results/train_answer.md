Brouwer's fixed-point theorem addresses a deceptively simple question: if you take a compact, convex, "filled" body in Euclidean space and apply any continuous map that keeps the body inside itself, must some point stay exactly where it was? The answer is yes. More formally, every continuous map f from a nonempty compact convex set K ⊂ ℝⁿ into itself has a point x* with f(x*) = x*. The one-dimensional case is elementary: g(x) = f(x) − x is nonnegative at one end of the interval and nonpositive at the other, so the intermediate value theorem forces a zero. That one-dimensional proof, however, relies on the ordering of the real line and cannot be lifted to higher dimensions, where f(x) − x is a vector that can swirl around without ever vanishing. Earlier approaches using the Kronecker index or the homology of the boundary sphere reach the same conclusion, but they require either differentiability or substantial algebraic-topology machinery. What is missing is a proof that works for arbitrary continuous maps using only a finite, elementary certificate.

The method that fills the gap is Brouwer's fixed-point theorem proved via Sperner's lemma. The central move is to reduce a topological existence problem to a combinatorial parity count on a triangulation. Work on the standard simplex Δⁿ = {x ∈ ℝⁿ⁺¹ : x_i ≥ 0, Σx_i = 1}. Suppose, for contradiction, that a continuous self-map f has no fixed point. Then at every x, some coordinate must decrease, because x and f(x) have the same coordinate sum and are not equal. Color each point by an index i such that f(x)_i < x_i. This rule is a legal Sperner labeling: the corner e_l can only be colored l, and a point on a face can only use the colors of that face's corners. Now triangulate Δⁿ into tiny simplices. Sperner's lemma guarantees an odd number of "rainbow" cells—cells whose n+1 vertices carry all n+1 colors. Take a sequence of such triangulations whose mesh goes to zero. By compactness, some sequence of rainbow cells converges to a single point x*. For each color i, the corresponding vertex v_i of the rainbow cell satisfies f(v_i)_i < (v_i)_i; passing to the limit gives f(x*)_i ≤ x*_i for every i. Since the two vectors have the same sum, every inequality must be an equality, so f(x*) = x*. This contradicts the no-fixed-point assumption, proving the theorem. The result then transfers to any nonempty compact convex body because every such body is homeomorphic to a closed ball, and the fixed-point property is preserved by homeomorphism.

Sperner's lemma itself is proved by induction on dimension. In dimension one, a Sperner-labeled segment must change colors an odd number of times, yielding an odd number of rainbow edges. For dimension n, count the facets of cells that carry exactly the colors {1, …, n}. A rainbow cell contributes one such facet, a cell using only those n colors contributes two, and all other cells contribute zero, so the total count has the same parity as the number R of rainbow cells. Counting the same facets by location, interior facets are shared by two cells and cancel modulo two, while boundary facets must lie on the face spanned by corners 1, …, n. That face is itself a Sperner-labeled (n−1)-simplex, so by the induction hypothesis it has an odd number of rainbow facets. Therefore R is odd. This parity argument is the finite, hand-checkable shadow of the degree obstruction on the boundary sphere.

```python
import numpy as np

def triangulate_simplex(m):
    """Regular m-subdivision of the 2-simplex in barycentric coordinates."""
    pts, idx = [], {}
    for i in range(m + 1):
        for j in range(m + 1 - i):
            k = m - i - j
            idx[(i, j, k)] = len(pts)
            pts.append(np.array([i, j, k], dtype=float) / m)
    pts = np.array(pts)
    tris = []
    for i in range(m + 1):
        for j in range(m + 1 - i):
            k = m - i - j
            if k == 0:
                continue
            a = idx[(i, j, k)]
            b = idx[(i + 1, j, k - 1)]
            c = idx[(i, j + 1, k - 1)]
            tris.append([a, b, c])
            if j > 0:
                d = idx[(i + 1, j - 1, k)]
                tris.append([a, d, b])
    return pts, np.array(tris)

def sperner_color(x, fx):
    """Choose an index i with fx_i < x_i (guaranteed if x != fx on the simplex)."""
    return int(np.where(fx < x - 1e-12)[0][0])

def find_rainbow_triangles(pts, tris, f):
    colors = np.array([sperner_color(p, f(p)) for p in pts])
    rainbow = [tri for tri in tris if len(set(colors[tri])) == 3]
    return np.array(rainbow), colors

def approx_fixed_point(f, m=64):
    pts, tris = triangulate_simplex(m)
    rainbow, _ = find_rainbow_triangles(pts, tris, f)
    if len(rainbow) == 0:
        return None
    return pts[rainbow[0]].mean(axis=0)

# Example continuous self-map of the 2-simplex.
def example_map(x):
    A = np.array([
        [0.2, 0.5, 0.3],
        [0.6, 0.1, 0.3],
        [0.2, 0.4, 0.4]
    ])
    y = A @ x
    return 0.5 * x + 0.5 * y  # stays inside the simplex

x_star = approx_fixed_point(example_map, m=32)
print("approx fixed point:", x_star)
print("f(x*):            ", example_map(x_star))
print("|f(x*) - x*|:     ", np.linalg.norm(example_map(x_star) - x_star))
```
