The problem is to find a notion of volume that belongs to the symmetry of a group rather than to any particular coordinate system. On a discrete group, counting measure is invariant under translation because every group element just permutes the points. On Euclidean space, Lebesgue measure is invariant under vector addition because translations preserve length and area. On a compact Lie group, an invariant differential form gives a normalized probability measure. But these are all special cases that rely on extra structure. A general topological group may have no discrete points to count, no vector-space chart, no smooth manifold structure, and no finite total size. What is needed is a construction that uses only the group operation and the topology.

The obstacle is that a purely algebraic group can be too wild for ordinary measure theory. Arbitrary subsets need not behave well, and a finitely additive notion of size is not enough to support integration. The right setting is a locally compact Hausdorff topological group, where local compactness supplies compact neighborhoods and finite subcovers, and the Hausdorff property keeps the topology well behaved. Even then, the natural first attempt, covering a compact set by translates of a small open neighborhood, only gives a coarse covering number. Covering numbers are subadditive, not additive, so they do not by themselves define a measure. Some limiting process is needed to extract a genuinely linear object from these rough geometric comparisons.

The method that resolves this is the Haar measure. It is the canonical left-invariant regular Borel measure on a locally compact Hausdorff topological group. The construction begins not with sets but with continuous compactly supported functions, which are smoother than indicator functions and better suited to a limiting argument. For two nonnegative compactly supported continuous functions f and g, with g not identically zero, one defines a covering gauge [f:g] as the infimum of the sums of coefficients needed to dominate f by a positive linear combination of left translates of g. This gauge is finite because the support of f is compact and g is positive on a neighborhood whose translates can cover that support. It is also automatically left invariant in the first argument, since translating f only translates the covering family.

To remove dependence on the choice of measuring bump g, fix a nonzero nonnegative compactly supported continuous function f_0 and form the normalized gauge I_g(f) = [f:g] / [f_0:g]. As the support of g shrinks toward the identity, continuity forces continuous functions to look nearly constant at the scale of g, and the subadditivity of covering numbers turns into genuine additivity in the limit. A cluster point of these normalized gauges gives a nonzero positive linear functional I on the space of compactly supported continuous functions, normalized so that I(f_0) = 1 and invariant under left translation in the sense that I(L_x f) = I(f), where (L_x f)(t) = f(x^{-1} t). This is the invariant integral promised by the group topology.

The Riesz representation theorem then converts the functional I into a regular Borel measure mu, finite on compact sets and satisfying I(f) = integral of f with respect to mu. The invariance of I implies that mu(xE) = mu(E) for every group element x and every Borel set E. The measure is nonzero because I(f_0) = 1, and every nonempty open set has positive mass because one can place a nonzero nonnegative compactly supported function inside it. This gives existence: every locally compact Hausdorff group carries a left-invariant Haar measure.

Uniqueness follows from the same compact-covering estimates. If mu and nu are two nonzero regular left-invariant Borel measures, then after fixing f_0 the ratios of integrals of any nonnegative test function f and f_0 must agree for both measures. The reason is that both integrals satisfy the same translation-invariant local covering inequalities, and the only freedom is the overall unit. Consequently nu = c mu for some positive scalar c. Thus the Haar measure is canonical up to the choice of a unit of volume.

One subtlety remains. A left-invariant measure need not be right invariant. For any group element a, the right translate E maps to mu(Ea) is again a left-invariant regular Borel measure, so by uniqueness it equals Delta(a)^{-1} mu(E) for some positive scalar. The map Delta from G to the positive reals is the modular function, a continuous group homomorphism. When Delta is identically one, the group is called unimodular, and the same Haar measure is both left and right invariant. Compact groups, discrete groups, and abelian groups are all unimodular, but the general theorem only guarantees left invariance.

The code below implements a finite-approximation illustration on a concrete compact group, the special orthogonal group SO(2), which is unimodular. It constructs an invariant probability measure by averaging a test function over a dense set of group elements sampled uniformly in the angle parameter. For a compact group, Haar measure can be normalized to total mass one, and the uniform angle average converges to the Haar expectation.

```python
import numpy as np


def so2_matrix(theta):
    """Return the 2x2 rotation matrix for angle theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def haar_sample_so2(n):
    """Sample n Haar-random elements of SO(2): angles uniform on [0, 2*pi)."""
    angles = np.random.uniform(0, 2 * np.pi, size=n)
    return np.stack([so2_matrix(a) for a in angles])


def haar_expectation_so2(f, n=100_000):
    """
    Approximate the Haar expectation of a function f: SO(2) -> R.
    On SO(2), normalized Haar measure is dtheta / (2*pi).
    """
    angles = np.random.uniform(0, 2 * np.pi, size=n)
    samples = np.array([f(so2_matrix(a)) for a in angles])
    return np.mean(samples)


def left_translate(f, g):
    """Return the left translate L_g f defined by (L_g f)(x) = f(g^{-1} x)."""
    g_inv = g.T  # inverse of an SO(2) rotation is its transpose
    def translated(x):
        return f(g_inv @ x)
    return translated


def demo_invariance():
    """Verify numerically that Haar expectation is invariant under left translation."""
    # A simple test function: top-left matrix entry
    def f(g):
        return float(g[0, 0])

    g0 = so2_matrix(1.3)  # arbitrary group element
    f_translated = left_translate(f, g0)

    np.random.seed(0)
    mean_f = haar_expectation_so2(f, n=200_000)
    mean_translated = haar_expectation_so2(f_translated, n=200_000)

    print(f"E[f]            = {mean_f:.6f}")
    print(f"E[L_g0 f]       = {mean_translated:.6f}")
    print(f"Expected value  = 0 (integral of cosine over full period)")
    print(f"Difference      = {abs(mean_f - mean_translated):.2e}")


if __name__ == "__main__":
    demo_invariance()
```
