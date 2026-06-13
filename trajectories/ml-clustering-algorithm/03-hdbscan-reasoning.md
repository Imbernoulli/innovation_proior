K-Means did what I traded for and showed me its own ceiling in the same numbers. On digits it rescued
the dead setting decisively — ARI 0.534, NMI 0.671, silhouette 0.139, all three off DBSCAN's floor
(0.0003 / 0.011 / −1.0) — because ten honest centroids in 64-D cannot collapse the way a global `eps`
did. On blobs it beat DBSCAN clearly, ARI 0.853 mean against 0.70, vindicating "convex isotropic
Gaussians are K-Means's exact model." But moons is where the cost landed exactly as I called it:
ARI 0.481, NMI 0.383 — barely better than a coin flip on a two-class problem — against DBSCAN's 0.972.
Nearest-centroid is a Voronoi tessellation; it *must* cut each half-moon with a straight bisector, so it
sliced the interleaving arcs the wrong way. So now I can see the real shape of the ladder: DBSCAN owns
non-convex moons and dies on high-D digits; K-Means owns convex blobs and digits and dies on non-convex
moons. Each is excellent-on-some, terrible-on-another, and on a geometric-mean aggregate that is exactly
the wrong profile. What I want is the method that keeps DBSCAN's two virtues — arbitrary shape and
explicit noise, which are what won moons — *without* DBSCAN's fatal weakness, the single global `eps`
that concentrated itself into nothing on digits and capped it on the varying-density blobs. The failure
to fix is precise and it is DBSCAN's, not K-Means's: one density level cannot describe a structure that
lives at many densities.

Let me start from why one `eps` is structurally wrong, not just badly tuned. Hartigan's picture makes
density-based clustering exact: there is an unknown density `f` behind the points; fix a level `lambda`,
take the level set `{x : f(x) ≥ lambda}`, and call each maximal connected component a cluster — noise is
everything below `lambda`, clusters can be any shape (they are connected components), and nothing is
forced into a group. DBSCAN estimates this with a single radius: core iff the `eps`-ball holds
`MinPts` points, clusters are connected runs of core points. But `eps` sets *one* global density level.
If the data has a tight knot and a diffuse cloud in the same picture — which is precisely the blobs
setting, `cluster_std` from 0.5 to 1.5 — there is no single `eps` that keeps both: tighten it to resolve
the knot and the cloud falls to noise; loosen it to hold the cloud and the knot fuses with whatever is
near. That is exactly the seed-to-seed spread I watched on DBSCAN's blobs (ARI 0.59–0.76), and on digits
it is total. One threshold for a multi-density object is a structural mismatch, not a tuning annoyance.

The escape: don't pick one `eps` — look at all of them. Vary `lambda` (equivalently `eps = 1/lambda`)
and watch the components evolve. As the level rises a component shrinks by shedding its fringe, splits
into two, or dies. That nested family over all levels is the density-contour tree, and it is the honest
object: the knot's cluster lives at high `lambda`, the cloud's at low `lambda`, and both are different
*nodes of one tree*. So the goal sharpens — build the whole tree of density-based clusterings at once,
then read a single flat partition off it by taking clusters from *different* levels rather than one
horizontal slice. A horizontal slice is what `eps` was; that is the thing that failed.

How do I build all DBSCAN clusterings for all `eps` without rerunning at a thousand radii? Look at what
the cluster relation actually is. Use the clean variant that keeps clusters as exactly the connected
components of core objects (drop the border-point subtlety, so a cluster is precisely a chunk of the
level set). Two core points `p, q` join at radius `eps` when a chain of core points links them, each
consecutive pair within `eps`; the atomic relation is "directly reachable at `eps`": `d(p,q) ≤ eps`, `p`
core, `q` core. Name the radius at which a point first becomes core: the **core distance** `d_core(p)`
is the distance to its `m_pts`-th nearest neighbor (counting `p` itself), so `|N_eps(p)| ≥ m_pts` exactly
when `eps ≥ d_core(p)`; its reciprocal `1/d_core(p)` is a `K`-NN density estimate with `K = m_pts`, which
is why `m_pts` is a density-smoothing knob, not a cluster-count knob. Then `p, q` are directly reachable
at `eps` iff `eps ≥ d_core(p)`, `eps ≥ d_core(q)`, and `eps ≥ d(p,q)` — all three at once iff
`eps ≥ max(d_core(p), d_core(q), d(p,q))`. Define that maximum as the **mutual reachability distance**
`d_mreach(p,q) = max(d_core(p), d_core(q), d(p,q))`. It is symmetric, reduces to ordinary distance when
both core distances are small, and *inflates* a pair when either endpoint is in a sparse region — a
sparse point cannot be closer than its own core distance to anything. That inflation is exactly the
defense against single-linkage chaining: sparse bridges become long edges while dense interiors stay as
they were. (OPTICS's reachability is the asymmetric one-sided version; taking the max over both core
distances removes the asymmetry that forces OPTICS's heuristic extraction.)

Now the payoff. Keeping mutual-reachability edges with weight `≤ eps` gives the same connectivity as
DBSCAN at that radius for *every* `eps` at once, because the edge condition already enforces both
endpoints core. Removing edges from a weighted graph in decreasing weight order while tracking connected
components *is* single-linkage hierarchical clustering. So running DBSCAN over every radius equals
running single-linkage once — after replacing ordinary distances by mutual reachability. And
single-linkage only needs the minimum spanning tree: the MST carries every edge that can ever be the
lightest connector between two components, hence the whole hierarchy. Compute the MST of the
mutual-reachability graph (Prim's, `O(n²)` dense), sort its edges, and reading the linkage backward —
remove edges in decreasing weight — gives the divisive splits. One conceptual patch: a lone surviving
point of a shrinking cluster must become noise once `eps` falls below its own `d_core`, which is not a
merge edge between two objects; extend the MST with a self-edge at each vertex weighted by its core
distance, so the hierarchy records both component splits and core-to-noise transitions.

The raw hierarchy is too large — most edge removals just drop one fringe point and are bookkeeping
noise, not real splits. So condense it with a **minimum cluster size** `m_clSize`. When an edge (or a
tied set) breaks a component, any resulting piece with fewer than `m_clSize` points is spurious. If two
or more non-spurious pieces remain, that is a true split — each large piece gets a new label; if exactly
one survives, the cluster merely *shrank* and keeps its label while the small pieces fall to noise; if
none survives, the cluster *died*. This turns the dendrogram into a compact tree of significant
components — a discrete approximation to Hartigan's density-contour tree.

I still need a flat clustering, and a horizontal cut is exactly what failed, so I choose clusters from
different levels. The right quantity is **prominence** — prefer a cluster whose points stay members
across a wide density interval. Plain lifetime is too crude (fringe and core points leave at different
levels); raw excess of mass is monotone along a branch and always biases toward ancestors. The fix is
**relative excess of mass**: for a cluster `C_i` born at level `lambda_min(C_i)`, each point `x_j`
contributes over the interval it belongs to `C_i` before leaving, `lambda_max(x_j, C_i) − lambda_min(C_i)`
(with `lambda = 1/eps`, each term is nonnegative: higher departure density minus lower birth density),
summed to the **stability** `S(C_i)`. Then extract the non-overlapping set of clusters maximizing total
stability — I cannot select both a node and a descendant, or a point gets two labels — by a bottom-up
dynamic program: `S_hat(C_i) = S(C_i)` at a leaf, else `max(S(C_i), Σ_children S_hat)`. If the children's
propagated sum strictly exceeds the node's own stability, the finer structure is more stable, deselect
the node and propagate the children up; otherwise the parent wins (ties to the parent) and its whole
subtree is suppressed. Exclude the root; uncovered points are noise. This is excess-of-mass (EOM)
selection, and it is precisely the thing DBSCAN could not do: a diffuse cluster and a dense cluster can
*both* appear in the final labeling because they are selected at different levels of the same tree.

Now the edit, and the scaffold decides how literal I get. `sklearn.cluster.HDBSCAN` realizes all of the
above — mutual-reachability single-linkage, size-based condensation, EOM extraction — so I do not
hand-roll the MST and the tree DP; I instantiate it. The two knobs that matter are `min_cluster_size`
(the `m_clSize` granularity) and `min_samples` (the `m_pts` density smoothing). I set
`min_cluster_size = max(5, n // 50)` — scale it with the dataset so "significant" means a couple of
percent of the points rather than a fixed count that would be too coarse on the 1797-point digits and
too fine on the 1000-point moons — and `min_samples = 5`, a small fixed smoothing, with
`cluster_selection_method = "eom"` for the excess-of-mass extraction I just derived. HDBSCAN determines
its own cluster count, so it ignores the harness's `n_clusters` — which is the whole point, the
self-determination DBSCAN had and K-Means lacked. But there is one harness-specific guard the abstract
method does not carry, and I keep it because the high-dimensional setting demands it: if HDBSCAN labels
*everything* noise — `len(set(labels)) ≤ 1`, the degenerate case the silhouette scores at −1.0 — I fall
back to `KMeans(k)` with the harness's `n_clusters`, so even if density estimation gives up in 64-D I
return `k` honest clusters instead of the collapse that killed DBSCAN on digits. That fallback is
exactly the lesson of step 2 folded in as a safety net: never hand the harness a degenerate labeling.
The full class is in the answer.

Reading K-Means's and DBSCAN's numbers together, here is what I expect and why it should be the
strongest rung. On **moons**, HDBSCAN should recover DBSCAN's win and likely exceed it — the
hierarchical, self-tuning density model finds the two arcs without a hand-set `eps`, so ARI should be at
or near 1.0, far above K-Means's 0.481 and at least matching DBSCAN's 0.972. On **blobs**, it should
beat DBSCAN's capped 0.70: selecting clusters at *different* density levels is the direct cure for the
varying-`cluster_std` problem that a single `eps` could not handle, so I expect ARI up around the
high-0.7s/low-0.9s, competitive with K-Means and with less of DBSCAN's seed spread. On **digits**, the
honest expectation is *non-degenerate but modest*: density estimation in 64-D is genuinely hard, so
HDBSCAN will likely leave many points as noise and recover only the cleanest digit clusters — I expect
ARI well above DBSCAN's 0.0003 but probably below K-Means's 0.534, with the KMeans fallback as insurance
against an outright collapse. The falsifiable claim against the prior rungs: HDBSCAN should be the only
method on the ladder that is non-degenerate on *all three* geometries at once — moons near-perfect,
blobs strong, digits real — so even where it trails K-Means on a single setting, its geometric-mean
aggregate should top both, because it is the first rung that refuses to throw away any geometry. If
digits comes in low enough to drag the gmean below K-Means's, that is the residual the ladder ends on:
the high-dimensional setting where neither density estimation nor a fixed-`k` partition is clearly right,
and where the genuinely better move would combine a density-hierarchy's shape sense with a
dimension-robust embedding — but within this harness's edit surface, the self-tuning multi-level density
model is the strongest single algorithm I can land.
