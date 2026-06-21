I present the canonical method as smoothed analysis of the simplex method, a framework that explains why Dantzig's simplex algorithm is fast in practice despite having exponential worst-case instances. The puzzle is old and sharp. On real linear programs the simplex method typically needs a number of pivot steps that grows only linearly with problem size, yet Klee and Minty's squashed cube and its successors force every standard deterministic pivot rule through exponentially many vertices. Worst-case analysis therefore labels the method intractable, while average-case analyses by Borgwardt, Smale, and Megiddo prove polynomial behavior only on idealized rotationally symmetric random inputs that look nothing like the structured problems practitioners solve. The two classical modes of analysis fail for opposite reasons, and neither captures the actual experience of running simplex on noisy real data.

The diagnosis is that worst-case analysis is blind to brittleness. A Klee-Minty cube is hard only because its facet tilts are specified to infinite precision; nudge any facet by a tiny amount and the exponentially long path collapses. Real data always carry such nudges, from measurement error, rounding, and finite precision. So what we want is an analysis that keeps the adversary, because the input may be chosen as maliciously as possible, but forbids the adversary from being infinitely precise. Smoothed complexity formalizes this by taking, for each input size n and perturbation magnitude sigma, the maximum over inputs of the expected running time after adding independent Gaussian noise of standard deviation sigma times the input magnitude. As sigma tends to zero this recovers worst-case analysis; as sigma grows large it approaches average-case analysis; and the meaningful middle regime is when the bound is polynomial in n and one over sigma. A bound polynomial in the logarithm of one over sigma would be too strong, because plugging in sigma of order two to the power negative L, where L is bit-length, would give a worst-case polynomial bound and collapse the notion back to ordinary polynomial time.

To make smoothed analysis tractable for the simplex method, the right pivot rule is shadow-vertex. Unlike other pivot rules, whose sequence of visited vertices is defined only by iteration, shadow-vertex produces its path as a closed-form geometric projection. Pick a starting objective t whose optimal vertex is known and consider the plane spanned by t and the true objective z. Project the feasible polyhedron onto this plane; the projection, called the shadow, is a convex polygon whose vertices and edges are images of actual polyhedron vertices and edges. Rotating the objective continuously from t toward z and tracking the optimal vertex walks around the boundary of this shadow polygon, so the number of pivots is at most the number of shadow edges. The entire running-time question collapses to bounding the expected size of this two-dimensional shadow under a Gaussian perturbation of the constraints.

The geometric heart of the result is a shadow-size theorem. For n Gaussian-perturbed constraints in d dimensions with centers of norm at most one and perturbation standard deviation sigma, the expected number of shadow edges is O of n d cubed over sigma to the sixth power. The proof reduces the count of shadow edges to the probability that the rotating objective lies within angle epsilon of the boundary of its current optimal simplex. If that probability is at most K times epsilon, then summing over a fine angular discretization gives an expected shadow size of order K. Establishing the linear-in-epsilon bound uses the Blaschke change of variables from integral geometry, rewriting the constraint vectors in terms of the normal and offset of their affine hull plus in-plane positions. The Jacobian of this change contains two revealing factors: the volume of the simplex, which rewards fat, non-degenerate bases and penalizes the sliver-like configurations that drive worst-case hardness, and the cosine of the angle between the hull normal and the objective.

The small-angle event then splits into a product of a distance factor and an incidence factor. The distance factor asks how close the facet's affine hull passes to the origin; because perturbed simplices are unlikely to be slivers, this has probability linear in epsilon with rate d squared over sigma to the fourth power. The incidence factor asks how close the facet normal is to being perpendicular to the objective; the n side constraints each shift as the plane tilts, giving a quadratic bound with rate n over sigma squared. A dyadic combination lemma turns the product of a linear and a quadratic bound back into a linear bound in epsilon, yielding the overall O of n d cubed over sigma to the sixth power shadow-size estimate. The brittle Klee-Minty configurations are exactly the ill-conditioned, near-boundary arrangements that the Gaussian perturbation renders exponentially improbable.

This shadow bound alone does not yet give a complete algorithm, because the projection plane depends on the data and a real implementation needs a feasible starting vertex, infeasibility detection, and positive right-hand sides. These issues are resolved by a two-phase construction. In the first phase, choose a well-conditioned random d-by-d minor by sampling several random d-subsets and keeping the one with largest smallest singular value; use it to build an auxiliary linear program LP prime with a known feasible basis and known starting objective. In the second phase, introduce an interpolation variable that morphs LP prime back into the original LP, and run shadow-vertex on the interpolated program. The perturbation is split into two independent Gaussians; the first fixes the projection plane, the second supplies the randomness for the shadow bound. Non-Gaussian ratio distributions that arise during interpolation are handled locally by density comparison, and the many-good-choices lemma guarantees that the start minor is well-conditioned with overwhelming probability. The final theorem states that the two-phase shadow-vertex method has polynomial smoothed complexity: the expected number of pivots is bounded by a polynomial in n, d, and one over sigma, capped by the trivial binomial worst-case bound. The optimized exponent is large, but the explanatory core is the clean shadow-size bound, which shows that small real-world noise is enough to destroy adversarial hardness.

The following Python script illustrates the central probabilistic claim. It samples Gaussian-perturbed linear programs, computes the optimal basis, and measures the smallest angle between the objective and any facet of the optimal basis's normal cone. The empirical probability that this angle is below epsilon grows roughly linearly in epsilon, matching the linear angle bound that underlies the shadow-size theorem.

```python
import numpy as np
from itertools import combinations


def optimal_basis(A, y, z):
    n, d = A.shape
    best_val, best_basis = -np.inf, None
    for I in combinations(range(n), d):
        AI = A[list(I), :]
        if np.linalg.matrix_rank(AI) < d:
            continue
        x = np.linalg.solve(AI, y[list(I)])
        if np.all(A @ x <= y + 1e-8):
            val = z @ x
            if val > best_val + 1e-8:
                best_val, best_basis = val, I
    return best_basis


def angle_to_cone_boundary(z, A, basis):
    A_I = A[list(basis), :]
    angs = []
    for j in range(A_I.shape[0]):
        rows = np.delete(A_I, j, axis=0)
        proj = rows.T @ np.linalg.solve(rows @ rows.T, rows @ z)
        cos = np.linalg.norm(proj) / (np.linalg.norm(z) + 1e-12)
        angs.append(np.arccos(np.clip(cos, 0, 1)))
    return min(angs)


def sample_angle(d, n, sigma, seed):
    rng = np.random.default_rng(seed)
    A_bar = rng.normal(0, 1, (n, d))
    A_bar = A_bar / np.maximum(np.linalg.norm(A_bar, axis=1, keepdims=True), 1.0)
    A = A_bar + rng.normal(0, sigma, A_bar.shape)
    y = np.full(n, 3.0) + rng.normal(0, sigma, n)
    z = rng.normal(0, 1, d)
    z = z / np.linalg.norm(z)
    basis = optimal_basis(A, y, z)
    return angle_to_cone_boundary(z, A, basis) if basis else None


if __name__ == "__main__":
    d, n, sigma = 3, 10, 0.05
    angles = [a for a in (sample_angle(d, n, sigma, i) for i in range(1000)) if a is not None]
    angles = np.array(angles)
    print(f"sampled {len(angles)} feasible instances")
    for eps in [0.002, 0.005, 0.01, 0.02, 0.05]:
        p = np.mean(angles < eps)
        print(f"eps={eps:.3f}: P(angle<eps)={p:.4f}, p/eps={p/eps:.1f}")
```
