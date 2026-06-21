We are given $n$ points $X = \{x_1, \dots, x_n\}$ in $\mathbb{R}^d$, no labels, and an integer $k$, and we want to carve them into $k$ groups so that within a group the points are tight and across groups they are far apart. Before searching for anything we need a single scalar that scores a grouping, because "tight" and "far apart" are intuitions, not a procedure. The cleanest summary of a group is a single representative point — its center — so "tight" becomes "every point is close to its representative," and a candidate set of $k$ centers $C = \{c_1, \dots, c_k\}$ is scored by the total squared distance from each point to its nearest center,

$$\phi(C) = \sum_{x \in X} \min_{c \in C} \| x - c \|^2 .$$

Choosing the centers already chooses the clustering: once $C$ is fixed, each point's natural group is the center it is closest to — the $\min$ literally picks it — so the whole problem collapses to choosing $k$ points $C$ that make $\phi$ small. Two facts make this hard. Minimizing $\phi$ exactly is NP-hard even for $k = 2$ in the plane, because moving the boundary between two groups changes both centroids, which moves the boundary again; an exact minimizer is off the table for any real $n$, and we are forced to a heuristic. And $\phi$, viewed as a function of the center locations, is non-convex — it is a pointwise minimum over centers of convex squared-distance pieces, a bumpy landscape with valleys of very different depths — so a local search can settle into a poor configuration with no a-priori bound on how poor. The pain is exactly this gap: a fast local search exists and is used everywhere, but its output quality is unpredictable and can be arbitrarily far from optimal.

The local search itself is forced by the structure of $\phi$, which tangles two kinds of variable — the center locations and the assignment of points to centers — so the move is to optimize one with the other held fixed and alternate. Fix the centers: each point contributes $\min_c \| x - c \|^2$, a minimum achieved by assigning the point to its nearest center, so the assignment step is exactly optimal term by term, and the resulting regions are Voronoi cells. Fix the assignment: the best center for a group $S$ is the $z$ minimizing $\sum_{x \in S} \| x - z \|^2$. Writing each $x - z = (x - c) + (c - z)$ with $c = c(S) = \frac{1}{|S|}\sum_{x \in S} x$ the center of mass, the cross term carries $\sum_{x \in S}(x - c) = 0$ by the very definition of the mean and vanishes, leaving the parallel-axis identity

$$\sum_{x \in S} \| x - z \|^2 = \sum_{x \in S} \| x - c(S) \|^2 + |S|\, \| c(S) - z \|^2 .$$

The first term is independent of $z$; the second is a nonnegative $|S|\,\|c(S)-z\|^2$ that is zero exactly when $z = c(S)$, so the unique minimizer is the arithmetic mean. This is also why the objective is squared rather than absolute: under squared error the optimal representative is the mean, a closed-form linear thing recomputed in one pass, whereas under absolute error it is the median, which has no such clean form. Both steps weakly decrease $\phi$; since $\phi \ge 0$ and there are at most $k^n$ partitions, the monotone march over a finite set terminates at a fixed point in finitely many sweeps — with no step size or learning rate to tune. But that fixed point is only a local minimum of $\phi$, and which basin we roll into is decided entirely by where the centers start. Uniform random seeding makes this acute: with five well-separated blobs and $k = 5$, uniform sampling is much more likely to draw two or three centers from one dense blob and leave another with none, and the loop happily converges to a clustering that merges two true groups and splits a third. The cost ratio $\phi/\phi_{\mathrm{OPT}}$ has no upper bound even with $n$ and $k$ fixed. The local search is fine; the seeding is the disease.

I propose k-means: the alternating Lloyd local search above, seeded by **k-means++**, a randomized $D^2$-weighted placement of the initial centers. What we want from the seeding is three things — the centers should be spread out (one per true group, not clumped), the placement should be robust (no single weird point should hijack it), and, the prize, it should come with a provable bound on the resulting $\phi$. A deterministic farthest-point rule — pick the first center, then repeatedly add the data point farthest from those chosen so far — does spread the centers, but "farthest point" is exactly where an outlier lives, so it preferentially plants centers on outliers; the spread is bought by sacrificing robustness completely. The fix for "always grab the single farthest" is to randomize: sample the next center with probability that grows with distance, so far-from-everything regions are strongly favored but any single point, even an outlier, carries only a small slice of the probability mass and cannot dominate. The question is which function of $D(x)$, the distance from $x$ to its nearest already-chosen center. Let the objective decide: $\phi = \sum_x D(x)^2$ once $D$ is distance-to-nearest, so each point's contribution to the very quantity we are shrinking is $D(x)^2$. Sampling proportional to $D(x)^2$ places a new center, with high probability, exactly where the current cost is concentrated. The seeding is therefore: first center uniform at random; each subsequent center sampled with probability

$$\frac{D(x_0)^2}{\sum_{x \in X} D(x)^2} ,$$

stopping at $k$. The square is not a tuning knob; it is the exponent matched to a squared-error objective.

What makes this worth the construction is that it is provably good. Compare against the optimal clustering $C_{\mathrm{OPT}}$ with potential $\phi_{\mathrm{OPT}}$, writing $\phi(A)$ and $\phi_{\mathrm{OPT}}(A)$ for a set $A$'s contributions. Seed a single optimal cluster $A$ uniformly at random: using the parallel-axis identity inside $E[\phi(A)] = \frac{1}{|A|}\sum_{a_0 \in A}\sum_{a \in A}\|a - a_0\|^2$, both resulting inner sums equal $A$'s spread about its own centroid, which is $\phi_{\mathrm{OPT}}(A)$, so $E[\phi(A)] = 2\,\phi_{\mathrm{OPT}}(A)$ — a random member sits at the cluster's RMS radius, which by the identity doubles the moment of inertia. For a $D^2$ seed that lands in $A$ with an arbitrary set of centers already in place, the new center is drawn proportional to current squared distance; bounding $D(a_0) \le D(a) + \|a - a_0\|$ (triangle), squaring with the power-mean inequality $(p+q)^2 \le 2p^2 + 2q^2$, and averaging over $a \in A$ so that the $\sum_a D(a)^2$ cancels the sampling denominator, both pieces collapse to $\frac{4}{|A|}\sum_{a_0}\sum_a \|a-a_0\|^2$, i.e. $E[\phi(A)] \le 8\,\phi_{\mathrm{OPT}}(A)$ — the factor 2 from triangle/power-mean times the factor 2 from the uniform-within-$A$ average. Finally, an induction over (centers left to place $t$, uncovered optimal clusters $u$) shows each new draw either spends itself in already-covered mass or converts one uncovered cluster to covered at expected cost $\le 8\,\phi_{\mathrm{OPT}}$ for that cluster, accumulating only a harmonic term:

$$E[\phi'] \le \big(\phi(X_c) + 8\,\phi_{\mathrm{OPT}}(X_u)\big)\,(1 + H_t) + \frac{u-t}{u}\,\phi(X_u), \qquad H_t = 1 + \tfrac12 + \dots + \tfrac1t .$$

Specializing to $t = u = k-1$ with the first (uniform) cluster as the only initially-covered one, and using $E[\phi(A)] = 2\,\phi_{\mathrm{OPT}}(A)$ for that first cluster and $H_{k-1} \le 1 + \ln k$, gives

$$E[\phi] \le 8(\ln k + 2)\,\phi_{\mathrm{OPT}} .$$

So the seeding alone, before the local search runs, lands within an $O(\log k)$ factor of the global optimum for any data set, with no separation assumption — and Lloyd's loop can only decrease $\phi$ from there, so the combined method inherits the guarantee. The randomization added to tame outliers is precisely what makes the bound provable, since the expectation is over the seeding's own randomness. The $\log k$ is genuinely there, not an artifact: a simplex-of-simplices instance forces $D^2$ seeding to be no better than $2\ln k$-competitive, so the upper bound is tight up to the constant. And the exponent is locked to the objective's: for $\phi^{[l]} = \sum_x \min_c \|x-c\|^l$ (with $l = 1$ the $k$-median objective), $D^l$ weighting gives $E[\phi^{[l]}] \le 2^{2l}(\ln k + 2)\,\phi_{\mathrm{OPT}}^{[l]}$; for the squared-error case $l = 2$ the centroid identity supplies the sharp constant 8.

A few practical wrinkles have to line up with the mathematics. The centroid step can produce an empty cluster whose mean is $0/0$; the repair is to relocate that dead center to the point with the largest $D^2$ — the same far-from-everything criterion as the seeding, spending the center where the present objective is large. A single $D^2$ draw can be unlucky, so at each seeding step we draw $2 + \lfloor \log k \rfloor$ candidates, compute for each the resulting potential $\sum_x \min(D(x)^2, \|x - \text{cand}\|^2)$, and greedily keep the one that reduces $\phi$ most — a logarithmic, lightweight variance hedge. Because the landscape is non-convex, the whole procedure is run $n_{\mathrm{init}} = 10$ times from independent seedings and the lowest-$\phi$ run is kept. The local loop is capped at $300$ iterations and stops on strict label stability or when the squared center shift falls below a tolerance; if it stops by center shift rather than exact label stability, one final assignment pass keeps the returned labels aligned with the returned centers.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


def _sq_dist_to_nearest(X, centers):
    """For each point: squared Euclidean distance to its nearest center, and which one."""
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)   # (n, k)
    return d2.min(axis=1), d2.argmin(axis=1)


def _kmeanspp_init(X, k, rng):
    """Seed k centers by D^2 weighting, greedily keeping the best of a few candidates."""
    n = X.shape[0]
    n_local_trials = 2 + int(np.log(k))
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    indices = np.full(k, -1, dtype=int)

    center_id = rng.choice(n)                              # first center: uniform at random
    centers[0] = X[center_id]
    indices[0] = center_id
    closest_d2 = ((X - centers[0]) ** 2).sum(axis=1)       # D(x)^2 to the chosen center
    current_pot = closest_d2.sum()                         # phi = sum_x D(x)^2

    for c in range(1, k):
        rand_vals = rng.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(np.cumsum(closest_d2), rand_vals)
        np.clip(candidate_ids, None, n - 1, out=candidate_ids)
        distance_to_candidates = ((X[candidate_ids, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        np.minimum(closest_d2, distance_to_candidates, out=distance_to_candidates)
        candidate_pot = distance_to_candidates.sum(axis=1)
        best = np.argmin(candidate_pot)                    # greedily keep the phi-reducing candidate
        current_pot = candidate_pot[best]
        closest_d2 = distance_to_candidates[best]
        best_candidate = candidate_ids[best]
        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    return centers, indices


def _centers_from_labels(X, labels, centers, d2):
    """Move nonempty clusters to their means; relocate empty clusters to far points."""
    k = centers.shape[0]
    new_centers = centers.copy()
    counts = np.bincount(labels, minlength=k)
    for j in range(k):
        if counts[j] > 0:
            new_centers[j] = X[labels == j].mean(axis=0)              # centroid step
    empty = np.where(counts == 0)[0]
    if len(empty) > 0:
        farthest = np.argsort(d2)[::-1]
        for j, point_id in zip(empty, farthest):
            new_centers[j] = X[point_id]                              # empty cluster relocation
    return new_centers


def _lloyd(X, centers, max_iter=300, tol=1e-4):
    """Batch Lloyd iteration with strict-label and center-shift stopping."""
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    strict_convergence = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        d2, labels = _sq_dist_to_nearest(X, centers)                  # assignment step
        new_centers = _centers_from_labels(X, labels, centers, d2)
        center_shift_tot = ((new_centers - centers) ** 2).sum()
        centers = new_centers

        if np.array_equal(labels, labels_old):
            strict_convergence = True
            break
        if center_shift_tot <= tol:
            break
        labels_old[:] = labels

    if not strict_convergence:
        d2, labels = _sq_dist_to_nearest(X, centers)                  # final E-step after tol/max_iter
    else:
        d2 = ((X - centers[labels]) ** 2).sum(axis=1)
    inertia = d2.sum()
    return centers, labels, inertia, n_iter


class CustomClustering(BaseEstimator, ClusterMixin):
    """k-means: k-means++ seeding + Lloyd refinement, best of n_init restarts."""

    def __init__(self, n_clusters=None, random_state=42, n_init=10, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        k = self.n_clusters if self.n_clusters is not None else 8
        if k > X.shape[0]:
            raise ValueError("n_clusters must be no larger than n_samples")
        rng = np.random.RandomState(self.random_state)
        best_inertia, best_centers, best_labels, best_n_iter = np.inf, None, None, 0
        for _ in range(self.n_init):                                  # non-convex: hedge with restarts
            centers, _ = _kmeanspp_init(X, k, rng)
            centers, labels, inertia, n_iter = _lloyd(X, centers, self.max_iter, self.tol)
            if inertia < best_inertia:                               # keep the lowest-phi run
                best_inertia = inertia
                best_centers, best_labels, best_n_iter = centers, labels, n_iter
        self.cluster_centers_, self.labels_ = best_centers, best_labels
        self.inertia_, self.n_iter_ = best_inertia, best_n_iter
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            self.fit(X)
        return _sq_dist_to_nearest(np.asarray(X, dtype=float), self.cluster_centers_)[1]
```
