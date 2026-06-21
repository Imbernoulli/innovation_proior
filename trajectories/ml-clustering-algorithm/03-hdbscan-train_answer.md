K-Means did what I traded for and showed me its ceiling in the same numbers. On digits it rescued the dead setting decisively — ARI $0.534$, NMI $0.671$, silhouette $0.139$, all off DBSCAN's floor — because ten honest centroids in 64-D cannot collapse the way a global $\mathrm{eps}$ did. On blobs it beat DBSCAN clearly, ARI $0.853$ mean against $0.70$. But moons landed exactly as I called it: ARI $0.481$, NMI $0.383$, barely better than a coin flip against DBSCAN's $0.972$, because nearest-centroid is a Voronoi tessellation that *must* cut each half-moon with a straight bisector. So the ladder's real shape is clear: DBSCAN owns non-convex moons and dies on high-D digits; K-Means owns convex blobs and digits and dies on moons. Each is excellent-on-some, terrible-on-another — exactly the wrong profile for a geometric mean. What I want keeps DBSCAN's two virtues, arbitrary shape and explicit noise, *without* its fatal weakness: the single global $\mathrm{eps}$ that concentrated into nothing on digits and capped the varying-density blobs.

I propose HDBSCAN — hierarchical density-based clustering with excess-of-mass extraction. Start from why one $\mathrm{eps}$ is structurally wrong, not just badly tuned. Hartigan's picture makes density-based clustering exact: there is an unknown density $f$ behind the points; fix a level $\lambda$, take the level set $\{x : f(x) \ge \lambda\}$, and call each maximal connected component a cluster, with noise everything below $\lambda$. DBSCAN estimates this with a single radius — core iff the $\mathrm{eps}$-ball holds $\mathrm{MinPts}$ points — so $\mathrm{eps}$ sets *one* global density level. If the data has a tight knot and a diffuse cloud in the same picture, which is precisely blobs with $\mathrm{cluster\_std}$ from $0.5$ to $1.5$, no single $\mathrm{eps}$ holds both: tighten it and the cloud falls to noise; loosen it and the knot fuses with its neighbor. That is the seed-to-seed spread I watched on DBSCAN's blobs, and on digits it was total. One threshold for a multi-density object is a structural mismatch.

The escape is to not pick one $\mathrm{eps}$ but look at all of them. Vary $\lambda$ (equivalently $\mathrm{eps} = 1/\lambda$) and watch the components evolve — as the level rises a component shrinks by shedding its fringe, splits, or dies. That nested family over all levels is the density-contour tree, and it is the honest object: the knot's cluster lives at high $\lambda$, the cloud's at low $\lambda$, both as different *nodes of one tree*. So the goal sharpens to building the whole tree of density-based clusterings at once, then reading a flat partition off it by taking clusters from *different* levels rather than one horizontal slice — a horizontal slice being exactly what $\mathrm{eps}$ was, the thing that failed.

To build all DBSCAN clusterings for all $\mathrm{eps}$ without rerunning at a thousand radii, look at what the cluster relation is. Use the clean variant where a cluster is exactly the connected components of core objects (dropping the border subtlety). Two core points $p, q$ are directly reachable at $\mathrm{eps}$ when $d(p,q) \le \mathrm{eps}$ and both are core. Name the radius at which a point first becomes core: the **core distance** $d_{\mathrm{core}}(p)$ is the distance to its $m_{\mathrm{pts}}$-th nearest neighbor (counting $p$ itself), so $|N_{\mathrm{eps}}(p)| \ge m_{\mathrm{pts}}$ exactly when $\mathrm{eps} \ge d_{\mathrm{core}}(p)$. Its reciprocal $1/d_{\mathrm{core}}(p)$ is a $K$-NN density estimate with $K = m_{\mathrm{pts}}$, which is why $m_{\mathrm{pts}}$ is a density-smoothing knob, not a cluster-count knob. Then $p, q$ are directly reachable at $\mathrm{eps}$ iff $\mathrm{eps} \ge d_{\mathrm{core}}(p)$, $\mathrm{eps} \ge d_{\mathrm{core}}(q)$, and $\mathrm{eps} \ge d(p,q)$ — all three at once iff $\mathrm{eps} \ge \max\bigl(d_{\mathrm{core}}(p), d_{\mathrm{core}}(q), d(p,q)\bigr)$. Define that maximum as the **mutual reachability distance**

$$ d_{\mathrm{mreach}}(p,q) = \max\bigl(d_{\mathrm{core}}(p),\, d_{\mathrm{core}}(q),\, d(p,q)\bigr). $$

It is symmetric, reduces to ordinary distance when both core distances are small, and *inflates* a pair when either endpoint is in a sparse region — a sparse point cannot be closer than its own core distance to anything. That inflation is exactly the defense against single-linkage chaining: sparse bridges become long edges while dense interiors stay as they were. (OPTICS's reachability is the asymmetric one-sided version; taking the max over both core distances removes the asymmetry that forces OPTICS's heuristic extraction.)

Now the payoff. Keeping mutual-reachability edges with weight $\le \mathrm{eps}$ gives the same connectivity as DBSCAN at that radius for *every* $\mathrm{eps}$ at once, because the edge condition already enforces both endpoints core. Removing edges from a weighted graph in decreasing weight order while tracking connected components *is* single-linkage hierarchical clustering. So running DBSCAN over every radius equals running single-linkage once — after replacing ordinary distances by mutual reachability — and single-linkage only needs the minimum spanning tree, which carries every edge that can ever be the lightest connector between two components. So I compute the MST of the mutual-reachability graph (Prim's, $O(n^2)$ dense), sort its edges, and read the linkage backward — removing edges in decreasing weight — to get the divisive splits. One conceptual patch: a lone surviving point of a shrinking cluster must become noise once $\mathrm{eps}$ falls below its own $d_{\mathrm{core}}$, which is not a merge edge between two objects, so I extend the MST with a self-edge at each vertex weighted by its core distance, recording both component splits and core-to-noise transitions.

The raw hierarchy is too large — most edge removals just drop one fringe point. So I condense it with a **minimum cluster size** $m_{\mathrm{clSize}}$. When an edge breaks a component, any resulting piece with fewer than $m_{\mathrm{clSize}}$ points is spurious. If two or more non-spurious pieces remain, that is a true split and each large piece gets a new label; if exactly one survives, the cluster merely *shrank* and keeps its label while the small pieces fall to noise; if none survives, the cluster *died*. This turns the dendrogram into a compact tree of significant components — a discrete approximation to Hartigan's density-contour tree.

I still need a flat clustering, and a horizontal cut is what failed, so I choose clusters from different levels. The right quantity is **prominence** — prefer a cluster whose points stay members across a wide density interval. Plain lifetime is too crude, and raw excess of mass is monotone along a branch and always biases toward ancestors. The fix is **relative excess of mass**: for a cluster $C_i$ born at level $\lambda_{\min}(C_i)$, each point $x_j$ contributes over the interval it belongs to $C_i$ before leaving, $\lambda_{\max}(x_j, C_i) - \lambda_{\min}(C_i)$ (with $\lambda = 1/\mathrm{eps}$, each term nonnegative — higher departure density minus lower birth density), summed to the **stability**

$$ S(C_i) = \sum_{x_j \in C_i} \bigl(\lambda_{\max}(x_j, C_i) - \lambda_{\min}(C_i)\bigr). $$

I then extract the non-overlapping set of clusters maximizing total stability — I cannot select both a node and a descendant, or a point gets two labels — by a bottom-up dynamic program: $\hat{S}(C_i) = S(C_i)$ at a leaf, else $\hat{S}(C_i) = \max\bigl(S(C_i), \sum_{\text{children}} \hat{S}\bigr)$. If the children's propagated sum strictly exceeds the node's own stability, the finer structure is more stable — deselect the node and propagate the children up; otherwise the parent wins (ties to the parent) and its whole subtree is suppressed. Exclude the root; uncovered points are noise. This is excess-of-mass (EOM) selection, and it is precisely what DBSCAN could not do: a diffuse cluster and a dense cluster both appear in the final labeling because they are selected at different levels of the same tree.

Since `sklearn.cluster.HDBSCAN` realizes all of this — mutual-reachability single-linkage, size-based condensation, EOM extraction — I instantiate it rather than hand-roll the MST and tree DP. The two knobs that matter are `min_cluster_size` (the $m_{\mathrm{clSize}}$ granularity) and `min_samples` (the $m_{\mathrm{pts}}$ density smoothing). I set $\mathrm{min\_cluster\_size} = \max(5,\, n // 50)$ — scaled with the dataset so "significant" means a couple of percent of the points rather than a fixed count that would be too coarse on the 1797-point digits and too fine on the 1000-point moons — and $\mathrm{min\_samples} = 5$, with `cluster_selection_method="eom"`. HDBSCAN determines its own cluster count, so it ignores the harness's `n_clusters` — the self-determination DBSCAN had and K-Means lacked. But I keep one harness-specific guard the abstract method does not carry, because the high-dimensional setting demands it: if HDBSCAN labels *everything* noise (`len(set(labels)) <= 1`, the degenerate case the silhouette scores at $-1.0$), I fall back to `KMeans(k)` with the harness's `n_clusters`, so even if density estimation gives up in 64-D I return $k$ honest clusters instead of the collapse that killed DBSCAN on digits. That fallback is step 2's lesson folded in as a safety net: never hand the harness a degenerate labeling.

Reading both prior rungs together, here is what I expect and why this should be strongest. On moons, HDBSCAN should recover DBSCAN's win and likely exceed it — the self-tuning density model finds the two arcs without a hand-set $\mathrm{eps}$, so ARI near $1.0$, far above K-Means's $0.481$. On blobs, it should beat DBSCAN's capped $0.70$, since selecting clusters at *different* density levels is the direct cure for varying $\mathrm{cluster\_std}$ — ARI in the high-$0.7$s to low-$0.9$s with less seed spread. On digits, the honest expectation is non-degenerate but modest: density estimation in 64-D is genuinely hard, so HDBSCAN will likely leave many points as noise and recover only the cleanest clusters — ARI well above DBSCAN's $0.0003$ but probably below K-Means's $0.534$, with the KMeans fallback as insurance. The claim against the prior rungs: HDBSCAN should be the only method on the ladder non-degenerate on *all three* geometries at once, so even where it trails K-Means on a single setting, its geometric-mean aggregate should top both — it is the first rung that refuses to throw away any geometry.

```python
class CustomClustering(BaseEstimator, ClusterMixin):
    """HDBSCAN — hierarchical density-based clustering (Campello et al., 2013)."""

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        from sklearn.cluster import HDBSCAN

        # HDBSCAN automatically determines the number of clusters.
        # min_cluster_size controls granularity.
        min_cs = max(5, X.shape[0] // 50)
        self._model = HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=5,
            cluster_selection_method="eom",
        )
        self._model.fit(X)
        self.labels_ = self._model.labels_

        # If HDBSCAN assigns everything to noise (-1), fall back to
        # labeling all points as cluster 0 to avoid degenerate metrics.
        if len(set(self.labels_)) <= 1:
            from sklearn.cluster import KMeans
            k = self.n_clusters if self.n_clusters is not None else 8
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(X)
            self.labels_ = km.labels_

        return self

    def predict(self, X):
        if self.labels_ is None:
            self.fit(X)
        return self.labels_


def custom_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))
```
