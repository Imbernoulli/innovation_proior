The problem is to decide when a sequence of geometric objects must have a limit, even though the objects may not share a common coordinate chart, parametrization, or ambient space. Classical compactness theorems are tied to a fixed background: Arzela-Ascoli needs a common domain and target, Sobolev compactness needs a fixed domain with uniform derivative bounds, and Hausdorff or varifold compactness needs a fixed ambient space. These tools break as soon as the right notion of "same point" is missing, or when domains reparametrize and energy concentrates. The real gap is therefore conceptual: compactness has been treated as convergence of maps, not as a property of an intrinsic family.

The method that fixes this is Gromov compactness. It abandons fixed coordinates and embeddings and instead compares compact metric spaces intrinsically, either by isometrically embedding them into a common metric space and measuring Hausdorff distance or, equivalently, by using a correspondence whose distortion of pairwise distances is small. This defines the Gromov-Hausdorff topology directly on metric spaces. The compactness hypothesis is then uniform total boundedness: for every epsilon, there is a uniform bound on the number of epsilon-balls needed to cover each space. A diameter bound plus these scale-by-scale covering bounds forces a subsequence to converge in the Gromov-Hausdorff sense. In Riemannian geometry, curvature and volume comparison estimates translate lower Ricci bounds, dimension bounds, and diameter bounds into exactly these covering estimates, so the abstract theorem becomes a theorem about families of manifolds. The limit may be singular or collapsed, but it remains a genuine metric object.

For pseudoholomorphic curves the same principle appears with energy as the controlling quantity. A sequence of J-holomorphic curves with a uniform energy bound may fail to converge as parametrized maps because energy can concentrate at points and bubbles can form. Gromov compactness says that, after allowing reparametrization and after enlarging the limit category to stable maps, a subsequence does converge: smoothly away from finitely many concentration points, with bubbles or nodes recording the energy that would otherwise vanish. Bounded energy means only finitely many bubbles above any positive threshold, so the extraction mechanism is a diagonal argument across energy scales. The theorem does not forbid degeneration; it classifies the degeneration that bounded energy permits.

Both versions share one idea: identify the natural intrinsic measurement of the family, prove uniform bounds on it, enlarge the notion of limit only as much as those bounds require, and extract a subsequence in that intrinsic compactification. Metric bounds give finite epsilon-nets and distance matrices; energy bounds give regions of bounded energy density plus finitely many concentration points. The implementation below illustrates the metric form: it represents compact metric spaces, builds epsilon-nets with uniformly bounded cardinality, compares their distance matrices, and extracts a convergent subsequence by a diagonal construction.

```python
import numpy as np
from itertools import permutations

class MetricSpace:
    """Finite metric space represented by its distance matrix."""
    def __init__(self, dist_matrix, name=""):
        self.d = np.asarray(dist_matrix, dtype=float)
        self.n = self.d.shape[0]
        self.name = name
        assert np.allclose(self.d, self.d.T) and np.all(np.diag(self.d) == 0)
        assert np.all(self.d >= 0)

    def epsilon_net(self, eps):
        """Greedy eps-net: returns indices of centers."""
        remaining = set(range(self.n))
        centers = []
        while remaining:
            c = min(remaining)
            centers.append(c)
            covered = {i for i in remaining if self.d[c, i] < eps}
            remaining -= covered
        return centers


def net_distance_matrix(space, eps):
    """Distance matrix of an eps-net of a metric space."""
    centers = space.epsilon_net(eps)
    return space.d[np.ix_(centers, centers)]


def matrix_distortion(A, B):
    """Minimum max-norm distance between A and B over all relabelings.
    Returns infinity if the matrices have different sizes."""
    A, B = np.asarray(A), np.asarray(B)
    if A.shape != B.shape:
        return float('inf')
    n = A.shape[0]
    best = float('inf')
    for perm in permutations(range(n)):
        Ap = A[np.ix_(perm, perm)]
        best = min(best, np.max(np.abs(Ap - B)))
    return best


def extract_convergent_subsequence(spaces, covering_bound):
    """Given a sequence of metric spaces with uniform covering bounds,
    extract a subsequence whose epsilon-net distance matrices are close
    at every scale. This is the computational skeleton of Gromov compactness.

    covering_bound(eps) should return the maximum number of eps-balls
    needed to cover any space in the family.
    """
    eps_scales = [2.0 ** (-k) for k in range(1, 6)]
    remaining = list(range(len(spaces)))

    for eps in eps_scales:
        max_size = covering_bound(eps)
        remaining = [i for i in remaining
                     if len(spaces[i].epsilon_net(eps)) <= max_size]
        if not remaining:
            raise ValueError(f"No spaces satisfy covering bound at eps={eps}")

        # Find a large subsequence whose eps-net matrices are pairwise close.
        groups = {}
        for i in remaining:
            mat = net_distance_matrix(spaces[i], eps)
            key = (round(mat.shape[0]), round(np.max(mat) * 10))
            groups.setdefault(key, []).append(i)

        # Keep the group with the most members at this scale.
        remaining = max(groups.values(), key=len)

    return remaining


# Example: a sequence of metric spaces converging to a 4-point metric space.
base = np.array([[0.0, 1.0, 1.0, 1.4],
                 [1.0, 0.0, 1.4, 1.0],
                 [1.0, 1.4, 0.0, 1.0],
                 [1.4, 1.0, 1.0, 0.0]])
sequence = []
for k in range(12):
    noise = np.random.RandomState(k).rand(4, 4) * 0.05
    noisy = base + (noise + noise.T) / 2
    np.fill_diagonal(noisy, 0.0)
    sequence.append(MetricSpace(noisy, name=f"perturbed_{k}"))

if __name__ == "__main__":
    def covering_bound(eps):
        return 4 if eps > 0.01 else 10

    subseq = extract_convergent_subsequence(sequence, covering_bound=covering_bound)
    print("Extracted indices:", subseq)
    for i in subseq:
        print(sequence[i].name)
```
