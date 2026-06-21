DBSCAN's numbers say exactly what I feared. On moons it was excellent — ARI $0.972$, NMI $0.944$ — vindicating the density-backbone story. On blobs it was respectable but plainly capped (ARI $0.59$–$0.76$ across seeds, mean $0.70$), the spread itself the tell that a single global $\mathrm{eps} = 0.22$ fits the tight blobs and starves the loose ones. But digits is the catastrophe and it is unambiguous: ARI $0.000302$, NMI $0.0107$, silhouette $-1.0$. That $-1.0$ is the harness's signature for a *degenerate* labeling: in 64 dimensions the $k$-distance curve flattened, the knee was meaningless, the derived $\mathrm{eps}$ came out too small, and DBSCAN swept essentially the whole set into noise. Global density thresholding did not survive the curse of dimensionality. On a geometric-mean aggregate a single collapsed setting is ruinous, so I need a rung that is at least *non-degenerate everywhere* — one that returns honest clusters on digits even if I give up some shape flexibility. The harness hands me the true cluster count as `n_clusters`; DBSCAN ignored it, and on digits that refusal is what killed it. The obvious move is to *use* it.

I propose K-Means — Lloyd's algorithm with k-means++ seeding. I want a method that uses $k$ and cannot collapse, so I partition into exactly $k$ groups by a compactness objective, accepting convex cells as the price of robustness. Summarize each group by one representative center $c$; "tight" becomes "every point is close to its representative." Score a candidate set of $k$ centers $C = \{c_1, \dots, c_k\}$ by the total squared distance from each point to its nearest center,

$$ \phi(C) = \sum_x \min_c \lVert x - c \rVert^2 . $$

Choosing the centers already chooses the clustering, since the $\min$ assigns each point to its closest center, so the whole problem collapses to picking $k$ centers that minimize $\phi$. Exact minimization is hopeless — even $k = 2$ in the plane is NP-hard, because moving the boundary moves both centroids, which moves the boundary again — so the game becomes how good a *local* solution I can reach.

$\phi$ tangles two kinds of variable, the center locations and the point-to-center assignment, and the move with tangled variables is to optimize one with the other fixed and alternate. Fix the centers: each point contributes $\min_c \lVert x - c \rVert^2$, minimized by assigning it to its nearest center — exactly optimal term by term, carving space into Voronoi cells. Fix the assignment: the best center for a group $S$ minimizes $\sum_{x \in S} \lVert x - z \rVert^2$. Compute it rather than assert it. Let $c = \frac{1}{|S|}\sum_{x \in S} x$ and write $x - z = (x - c) + (c - z)$; expanding,

$$ \sum_{x\in S} \lVert x - z \rVert^2 = \sum_{x\in S} \lVert x - c \rVert^2 + 2(c - z)\cdot\!\sum_{x\in S}(x - c) + |S|\,\lVert c - z \rVert^2 . $$

The cross term carries $\sum_{x\in S}(x - c) = |S|c - |S|c = 0$ by the definition of the mean, so it vanishes, leaving $\sum \lVert x - z \rVert^2 = \sum \lVert x - c \rVert^2 + |S|\,\lVert c - z \rVert^2$. This is the parallel-axis identity: the first term is independent of $z$, the second is nonnegative and zero only at $z = c$, so the unique minimizer is the arithmetic mean. Now the squared distance pays off — under squared error the optimal representative is the *mean*, a closed-form one-pass quantity, whereas under absolute error it would be the median, which has no such linear form. The square is what makes the centroid step trivial.

Does the loop converge? The assignment step moves each point to its cost-minimizing center, so $\phi$ cannot increase; the centroid step moves each center to its group's squared-error minimizer, so again $\phi$ cannot increase. $\phi$ is monotonically non-increasing, bounded below by zero, and the loop marches through *partitions*, of which there are finitely many — a monotone sequence over a finite set must terminate at a fixed point where every point sits with its nearest center and every center is its group's mean. No step size, no learning rate, and crucially no global radius to mis-set: on digits, where DBSCAN's single $\mathrm{eps}$ produced nothing, this loop *cannot* collapse to one cluster — it is constructed to return exactly $k = 10$ non-empty groups, which alone should lift digits off the $-1.0$ floor.

But the fixed point is only a *local* minimum. $\phi$ as a function of the center locations is a pointwise minimum of convex squared-distance pieces — a bumpy non-convex landscape — and "never increases" only guarantees I roll into whatever basin I started in. Which valley is decided by the *initialization*. With the naive start, $k$ centers drawn uniformly from the data, picture five well-separated blobs and $k = 5$: uniform sampling very likely lands two or three centers in one dense blob and none in another, the empty blob gets swallowed and the over-seeded one gets split, and the loop converges to a clustering that merges two true groups and splits a third — with no upper bound on how bad, an unbounded $\phi / \phi_{\mathrm{OPT}}$ ratio with non-negligible probability. The local search is fine; the *seeding* is the disease, and I cannot afford a bad basin where blobs has five clusters of *varying* density, since the loose blob is exactly the one a clumped seeding starves.

What I want from seeding is one-per-group spread, robustness against a single weird point hijacking a center, and ideally a provable bound. Deterministic farthest-point seeding — first center somehow, then repeatedly the point farthest from all chosen centers — spreads, but "farthest point" *is* the most extreme outlier, so it plants centers on outliers; the spread is bought by sacrificing robustness. So keep the *tendency* of farthest-point without the brittleness: randomize it. Let $D(x)$ be the distance from $x$ to its nearest chosen center and sample the next center with probability proportional to an increasing function of $D(x)$ — far-from-everything regions are favored, but any single point, even an outlier, carries only a small slice of the probability mass, while a dense under-covered cluster has many moderately-far points whose mass sums. Which function? Let the objective decide: $\phi = \sum_x D(x)^2$, so each point's contribution to the very quantity I am shrinking is $D(x)^2$. Sample proportional to $D(x)^2$ — $D^2$ weighting — and I place each new center where the current cost is concentrated. The exponent is not a knob; it matches the squared-error objective. This is k-means++ seeding, and it earns the guarantee the bare loop lacks: a uniformly-seeded optimal cluster costs $2\,\phi_{\mathrm{OPT}}(A)$ in expectation by the parallel-axis identity, a $D^2$-seed landing in $A$ costs $\le 8\,\phi_{\mathrm{OPT}}(A)$ via triangle plus the power-mean inequality, and an induction over (centers left to place, clusters still uncovered) accumulates a harmonic factor to

$$ \mathbb{E}[\phi] \le 8(\ln k + 2)\,\phi_{\mathrm{OPT}} $$

for *any* data with no separation assumption — and the subsequent local search only lowers $\phi$. An $O(\log k)$ guarantee for the whole procedure, exactly where uniform seeding had none.

Since `scikit-learn` is in scope, `sklearn.cluster.KMeans` *is* this method: Lloyd's two forced steps with `init="k-means++"` as its default seeding (the $D^2$ construction above) and `n_init` independent restarts keeping the lowest-inertia run — the standard hedge against the non-convex landscape, since a single seeding can still be unlucky. So I instantiate `KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)`, where $k = \mathrm{n\_clusters}$ from the harness (8 only as a fallback that will not trigger). `n_init = 10` is the restart count that makes the $O(\log k)$ seeding robust in practice; `max_iter = 300` caps the local loop, which converges far sooner on these sizes; `predict` delegates to the fitted model's nearest-center rule.

Reading DBSCAN's numbers, here is what I expect and where this must lose. On digits — DBSCAN's dead setting — this should be a large, decisive gain: ten honest centroids in 64-D, no collapse, ARI and NMI well off the floor and silhouette positive. On blobs, K-Means should *beat* DBSCAN clearly, since convex isotropic Gaussians are its exact model and $D^2$ seeding plus restarts handle the varying $\mathrm{cluster\_std}$ far better than a single global $\mathrm{eps}$ — I expect ARI up around $0.85$ with less seed spread. The honest cost is moons: nearest-centroid is a Voronoi tessellation that *must* cut each half-moon with a straight bisector, so K-Means will slice the moons the wrong way and score far below DBSCAN's $0.972$, likely ARI under $0.5$. That is the trade made with eyes open — I sacrifice the one setting DBSCAN owned to stop *failing* the setting DBSCAN destroyed, and because the task aggregates by geometric mean, trading a moons collapse for a digits collapse is the right direction. If moons falls all the way to a poor convex split, that failure is itself the next rung's target, because the ideal method would keep DBSCAN's non-convex moons *and* K-Means's non-degenerate digits at once.

```python
class CustomClustering(BaseEstimator, ClusterMixin):
    """K-Means clustering (Lloyd's algorithm)."""

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self._model = None

    def fit(self, X):
        from sklearn.cluster import KMeans

        k = self.n_clusters if self.n_clusters is not None else 8
        self._model = KMeans(
            n_clusters=k, random_state=self.random_state, n_init=10, max_iter=300
        )
        self._model.fit(X)
        self.labels_ = self._model.labels_
        return self

    def predict(self, X):
        if self._model is None:
            self.fit(X)
        return self._model.predict(X)


def custom_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))
```
