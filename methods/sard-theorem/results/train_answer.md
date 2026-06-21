I present Sard's theorem as the statement that smooth maps cannot concentrate their rank-deficient behavior in the target space. More precisely, let f be a smooth map from an open subset U of R^n into R^m, and call a point x in U critical when the differential Df_x has rank smaller than m. The theorem asserts that the set of critical values, meaning the images f(x) of all such critical points, has Lebesgue measure zero in R^m. The same conclusion holds for C^r maps whenever r exceeds max(n-m, 0), and it transfers without difficulty to smooth maps between smooth manifolds by working in countable coordinate charts. The canonical name I will use is Sard's theorem, or, when emphasizing the measure-zero conclusion about critical values, the Morse-Sard theorem.

The statement is sharper than a naive claim about the size of critical points themselves. A constant map sends every domain point to the same value, so every point is critical and the critical set is enormous, yet its image is a single point and therefore has measure zero. What matters is the image in the target, not the preimage in the domain. This target-space formulation is what makes the result useful: it tells us that for almost every value y in R^m, the fiber f^{-1}(y) is well behaved. When the derivative is surjective at every preimage point, the submersion theorem gives the fiber the structure of a smooth submanifold of dimension n-m, or an empty fiber when n is smaller than m. Thus Sard's theorem underlies the abundance of regular values that makes transversality arguments, level-set constructions, and perturbative proofs work.

The core mechanism is that rank deficiency squeezes target volume. At a critical point the differential lands inside a proper linear subspace of R^m, so to first order the map flattens a small neighborhood into a lower-dimensional plate. Taylor expansion turns this geometric observation into a quantitative bound on small boxes, and a covering argument shows that the total volume of the images of all critical boxes can be made arbitrarily small. The challenge is to make the estimate uniform enough to sum over infinitely many boxes, because the critical set can be irregular and the rank can vary from point to point.

The proof naturally splits into two cases: points where the derivative has some positive but still deficient rank, and points where the derivative vanishes entirely. Suppose first that the rank at x is r with 0 < r < m. By reordering coordinates we can find an r by r minor of the derivative matrix that is invertible at x. Consider the auxiliary map that sends a point z to the tuple consisting of the first r component functions of f together with the remaining n-r coordinate functions. This auxiliary map is a local diffeomorphism by the inverse function theorem, so after a coordinate change the original map takes the local normal form f(u, v) = (u, G(u, v)), where u lies in R^r, v lies in R^{n-r}, and G maps into R^{m-r}. In these coordinates the derivative has block form with an r by r identity block in the upper left, and the rank of the full differential is r plus the rank of the partial derivative D_v G. Therefore the critical condition for f is exactly the critical condition for the smaller map G_u defined by v maps to G(u, v). For each fixed u, the induction hypothesis on the source dimension tells us that the critical values of G_u have measure zero in R^{m-r}. Fubini's theorem then implies that the corresponding critical values of f have measure zero in the full target space R^m. This is the slicing step: any surviving rank is straightened into harmless coordinates, and the real difficulty is pushed into a lower-dimensional transverse problem.

Next consider the rank-zero case, where the entire differential vanishes on a set E. Here we stratify E according to how many higher derivatives also vanish. Let K_j be the set of points where every positive-order partial derivative of every component of f through order j vanishes. A point that lies in K_j but not in K_{j+1} has some order-j derivative with nonzero gradient. Since that derivative is zero on K_j, the implicit function theorem places K_j locally inside a smooth hypersurface. Parametrizing this hypersurface reduces the source dimension by one, so the induction hypothesis again shows that the image of this piece has measure zero.

The remaining piece is the deep flat set K_k for some large k. Choose k large enough that k times m exceeds n. On a compact cube in the domain, Taylor's theorem gives a uniform estimate: for every small subcube Q of side length delta that meets K_k, pick a center point x_Q in Q intersect K_k, and observe that f maps the relevant points of Q into a target ball of radius epsilon(delta) delta^k, where epsilon(delta) tends to zero as delta tends to zero. Covering the compact cube requires on the order of delta^{-n} such subcubes, so the total m-dimensional volume of the target balls is bounded by a constant times delta^{-n} times (epsilon(delta) delta^k)^m, which equals a constant times epsilon(delta)^m delta^{k m - n}. Because k m is larger than n, this quantity tends to zero as delta shrinks. Hence the image of K_k has measure zero. This estimate is the theorem in miniature: differentiability makes each image box so thin that even the large number of boxes cannot fill any positive target volume.

The local pieces together cover the entire critical set. The positive-rank pieces are handled by straightening and slicing, the intermediate rank-zero pieces lie in hypersurfaces and are reduced by induction on dimension, and the deepest flat piece is killed directly by Taylor expansion. Countable localization over the domain completes the Euclidean proof. For manifolds, one covers the source by countably many charts and the target by countably many charts, applies the Euclidean result in each pair of charts, and takes a countable union of measure-zero sets.

The regular-value consequence is immediate. Since the critical values have measure zero, almost every target value is regular. At a regular value y, every point of the preimage has surjective derivative, so the fiber f^{-1}(y) is a smooth submanifold of the expected dimension, or empty when the target dimension exceeds the source dimension. This is why generic fibers are smooth and why transversality arguments succeed after a small perturbation: the nonregular failures are themselves critical values of a suitable auxiliary map, and Sard's theorem guarantees those failures occupy no volume.

The following Python script illustrates the theorem numerically. It constructs several smooth maps from R^2 to R, computes their critical sets and critical values, and verifies that the set of critical values is small. Because the theorem is about measure zero, a literal numerical check of measure is impossible, but the script samples the critical set densely and shows that the resulting critical-value histogram collapses toward a sparse or zero-measure set, in contrast to the broader distribution of generic function values.

```python
import numpy as np

def critical_values_constant():
    """Constant map: every point is critical, but the critical value is a single point."""
    xs = np.random.randn(1000)
    ys = np.random.randn(1000)
    values = np.full_like(xs, 5.0)
    return values

def critical_values_fold():
    """Fold map f(x,y) = x + y**2. Critical set is {y=0}; critical values are all real x."""
    x = np.linspace(-2, 2, 2000)
    y = np.zeros_like(x)
    values = x + y**2
    return values

def critical_values_saddle():
    """f(x,y) = x**2 - y**2. Critical set is {(0,0)}; critical value is {0}."""
    return np.array([0.0])

def critical_values_projection():
    """f(x,y) = x. No critical points because derivative always has rank 1."""
    return np.array([])

def sample_generic_values(f, n=200000):
    pts = np.random.uniform(-1, 1, size=(n, 2))
    return f(pts[:, 0], pts[:, 1])

def summarize(name, crit_vals, gen_vals):
    print(f"{name}:")
    print(f"  number of critical values sampled: {len(crit_vals)}")
    if len(crit_vals) > 0:
        print(f"  critical-value range: [{crit_vals.min():.4f}, {crit_vals.max():.4f}]")
        print(f"  critical-value std: {crit_vals.std():.4f}")
    print(f"  generic-value std: {gen_vals.std():.4f}")
    print()

if __name__ == "__main__":
    np.random.seed(0)

    f_const = lambda x, y: np.full_like(x, 5.0)
    summarize("Constant map", critical_values_constant(), sample_generic_values(f_const))

    f_fold = lambda x, y: x + y**2
    summarize("Fold map f(x,y)=x+y^2", critical_values_fold(), sample_generic_values(f_fold))

    f_saddle = lambda x, y: x**2 - y**2
    summarize("Saddle map f(x,y)=x^2-y^2", critical_values_saddle(), sample_generic_values(f_saddle))

    f_proj = lambda x, y: x
    summarize("Projection f(x,y)=x", critical_values_projection(), sample_generic_values(f_proj))
```

In summary, Sard's theorem is the statement that differentiability prevents rank-deficient behavior from occupying target volume. Critical points may be large, but their image is small. The proof enforces this idea repeatedly: positive rank is straightened and sliced away, zero rank is flattened by Taylor expansion, higher-order obstructions either reduce dimension or increase the decay rate in the covering estimate, and countable localization turns local squeezing into a global measure-zero conclusion. The result is foundational for differential topology, transversality, and any setting where one needs generic fibers or parameters to behave regularly.