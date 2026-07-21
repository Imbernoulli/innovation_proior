K-Means did what I traded for and showed me its own ceiling in the same numbers. On digits it rescued
the dead setting decisively — ARI 0.534, NMI 0.671, silhouette 0.139, all three off DBSCAN's floor
(0.0003 / 0.011 / −1.0) — because ten honest centroids in 64-D cannot collapse the way a global `eps`
did. On blobs it beat DBSCAN clearly, ARI 0.853 mean against 0.70, vindicating "convex isotropic
Gaussians are K-Means's exact model." But moons is where the cost landed exactly as I called it:
ARI 0.481, NMI 0.383 — barely better than a coin flip on a two-class problem — against DBSCAN's 0.972.
Nearest-centroid is a Voronoi tessellation; it *must* cut each half-moon with a straight bisector, so it
sliced the interleaving arcs the wrong way.

Let me put both prior methods' tables side by side on the actual scoring rule before I decide anything,
because the aggregate is a geometric mean and I want the shape of these results in numbers, not impressions. K-Means's
three setting-means are blobs `(0.853 + 0.874 + 0.585)/3 ≈ 0.771`, moons `(0.481 + 0.383 + 0.494)/3 ≈
0.453`, and digits `(0.534 + 0.671 + 0.139)/3 ≈ 0.448`, whose geometric mean is `(0.771 · 0.453 ·
0.448)^{1/3} = (0.1565)^{1/3} ≈ 0.54`. That 0.54 is finite and real — nothing degenerate — and it is the
number I now have to beat. DBSCAN's aggregate, by contrast, had a digits setting-mean of about `−0.33`
(the −1.0 silhouette dragging it negative), which makes its geometric mean ill-defined — degenerate —
so on the scoring rule DBSCAN is *below* K-Means despite owning moons outright. So the shape is
crisp: DBSCAN owns non-convex moons and dies on high-D digits; K-Means owns convex blobs and digits and
dies on non-convex moons. Each is excellent-on-some, terrible-on-another, and on a geometric-mean
aggregate that is exactly the wrong profile — the mean is pulled toward its *weakest* factor, so both
methods are held down, K-Means by its moons 0.453 and (would-be) DBSCAN by its digits −0.33.

A second signal in the tables shapes what "winning moons" can even look like. On the two moons rows,
DBSCAN's near-perfect ARI 0.972 scored silhouette only 0.224, while K-Means's *wrong* ARI 0.481 scored
silhouette 0.494 — higher. The intrinsic metric *prefers* the incorrect convex split, because two
interleaving crescents sit close in Euclidean space and a compact left/right cut looks more separated
than the true arcs. So whatever recovers the correct arcs will show its moons gain in ARI and NMI while
its moons silhouette stays low near 0.22 — I will not read that modest silhouette as a failure later; it
is baked into the geometry.

So what I want is the method that keeps DBSCAN's two virtues — arbitrary shape and explicit noise, which
are what won moons — *without* DBSCAN's fatal weakness, the single global `eps` that concentrated itself
into nothing on digits and capped it on the varying-density blobs. The failure to fix is precise and it
is DBSCAN's, not K-Means's: one density level cannot describe a structure that lives at many densities.
Let me first make sure that is a *structural* statement and not just "tune `eps` better," and then walk
the honest options for fixing it.

Hartigan's picture makes density-based clustering exact: there is an unknown density `f` behind the
points; fix a level `lambda`, take the level set `{x : f(x) ≥ lambda}`, and call each maximal connected
component a cluster — noise is everything below `lambda`, clusters can be any shape (they are connected
components), and nothing is forced into a group. DBSCAN estimates this with a single radius: core iff the
`eps`-ball holds `MinPts` points, clusters are connected runs of core points. But `eps` sets *one* global
density level. If the data has a tight knot and a diffuse cloud in the same picture — which is precisely
the blobs setting, `cluster_std` from 0.5 to 1.5 — there is no single `eps` that keeps both: tighten it
to resolve the knot and the cloud falls to noise; loosen it to hold the cloud and the knot fuses with
whatever is near. That is exactly the seed-to-seed spread I watched on DBSCAN's blobs (ARI 0.59–0.76),
and on digits it is total. One threshold for a multi-density object is a structural mismatch, not a
tuning annoyance.

The options. Grid-searching `eps` is no escape: with no labels at fit time I cannot pick "best," and
every run is still one horizontal density level, so a grid of horizontal cuts still cannot place the
knot's cluster and the cloud's cluster in the *same* output. K-Means again is guaranteed non-degenerate
but throws moons away — the method I just built and am here to avoid. The direction that works is a
*local* radius, each region declaring its own density scale. OPTICS is the classical version, but
extracting a flat clustering from its reachability plot needs a steepness threshold `ξ` — a new global
knob — forced by its *asymmetric* one-sided reachability. The local-radius idea is right; I want it with
the knob removed by construction.

Don't pick one `eps` — look at all of them. Vary `lambda` (equivalently `eps = 1/lambda`) and watch the
components evolve. As the level rises a component shrinks by shedding its fringe, splits into two, or
dies. That nested family over all levels is the density-contour tree, and it is the honest object: the
knot's cluster lives at high `lambda`, the cloud's at low `lambda`, and both are different *nodes of one
tree*. So the goal sharpens — build the whole tree of density-based clusterings at once, then read a
single flat partition off it by taking clusters from *different* levels rather than one horizontal slice.
A horizontal slice is what `eps` was; that is the thing that failed.

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
`eps ≥ max(d_core(p), d_core(q), d(p,q))`; the binding constraint is whichever of the two sparsities or
the raw gap is largest, so a pair in which one endpoint is sparse only links at that endpoint's larger
core distance, not the deceptively small raw gap. Define that maximum as the **mutual reachability
distance** `d_mreach(p,q) = max(d_core(p), d_core(q), d(p,q))`. It is symmetric, reduces to ordinary
distance when both core distances are small, and *inflates* a pair when either endpoint is in a sparse
region — a sparse point cannot be closer than its own core distance to anything. That inflation is
exactly the defense against single-linkage chaining, and a concrete number shows it does the work.
Picture two dense clusters bridged by a thin line of noise. A cluster point sits among neighbors at
spacing `~0.1` and reaches its `m_pts`-th neighbor at `d_core ≈ 0.05`; a bridge point in the sparse line
reaches its `m_pts`-th neighbor only far away, say `d_core ≈ 0.5`. The raw distance from the bridge point
to a cluster point might be `0.1`, small enough that single-linkage on raw distance would happily merge
across it. But `d_mreach(bridge, cluster) = max(0.5, 0.05, 0.1) = 0.5`, while a within-cluster edge is
`max(0.05, 0.05, 0.1) = 0.1` — the bridge edge is now five times heavier than the interior edges, so it
is the *last* thing a decreasing-weight linkage keeps and the *first* thing it drops. The sparse bridge
became a long edge while dense interiors stayed as they were, which is precisely single-linkage's
chaining disease cured at the metric level. (OPTICS's reachability is the asymmetric one-sided version;
taking the max over both core distances removes the asymmetry that forced its heuristic extraction — the
knob I set out to delete.)

Now the payoff. Keeping mutual-reachability edges with weight `≤ eps` gives the same connectivity as
DBSCAN at that radius for *every* `eps` at once, because the edge condition already enforces both
endpoints core. Removing edges from a weighted graph in decreasing weight order while tracking connected
components *is* single-linkage hierarchical clustering. So running DBSCAN over every radius equals
running single-linkage once — after replacing ordinary distances by mutual reachability. And
single-linkage only needs the minimum spanning tree: the MST carries every edge that can ever be the
lightest connector between two components, hence the whole hierarchy. Compute the MST of the
mutual-reachability graph (Prim's, `O(n²)` dense — for the task's `n ≤ 1797` that is about 3.2 million
edge relaxations, entirely affordable), sort its edges, and reading the linkage backward — remove edges
in decreasing weight — gives the divisive splits. One conceptual patch: a lone surviving point of a
shrinking cluster must become noise once `eps` falls below its own `d_core`, which is not a merge edge
between two objects; extend the MST with a self-edge at each vertex weighted by its core distance, so the
hierarchy records both component splits and core-to-noise transitions.

The raw hierarchy is too large — most edge removals just drop one fringe point and are bookkeeping
noise, not real splits. So condense it with a **minimum cluster size** `m_clSize`. When an edge (or a
tied set) breaks a component, any resulting piece with fewer than `m_clSize` points is spurious. If two
or more non-spurious pieces remain, that is a true split — each large piece gets a new label; if exactly
one survives, the cluster merely *shrank* and keeps its label while the small pieces fall to noise; if
none survives, the cluster *died*. This turns the dendrogram into a compact tree of significant
components — a discrete approximation to Hartigan's density-contour tree. I want `m_clSize` to mean "a
couple of percent of the data" rather than a fixed count that would be too coarse on the 1797-point
digits and too fine on the 1000-point moons, so I scale it: `m_clSize = max(5, n // 50)`. That is 35 on
digits, 30 on blobs, 20 on moons — a consistent ~2% floor on what counts as a cluster across the three
sizes.

I still need a flat clustering, and a horizontal cut is exactly what failed, so I choose clusters from
different levels. The right quantity is **prominence** — prefer a cluster whose points stay members
across a wide density interval. Plain lifetime is too crude (fringe and core points leave at different
levels); raw excess of mass is monotone along a branch and always biases toward ancestors. The fix is
**relative excess of mass**: for a cluster `C_i` born at level `lambda_min(C_i)`, each point `x_j`
contributes over the interval it belongs to `C_i` before leaving, `lambda_max(x_j, C_i) − lambda_min(C_i)`
(with `lambda = 1/eps`, each term is nonnegative: higher departure density minus lower birth density),
summed to the **stability** `S(C_i)`. Then extract the non-overlapping set of clusters maximizing total
stability — I cannot select both a node and a descendant, or a point gets two labels — by a bottom-up
dynamic program: `S_hat(C_i) = S(C_i)` at a leaf, else `max(S(C_i), Σ_children S_hat)`. When the children's
propagated stabilities outsum the parent's own the parent is deselected and the finer split is kept;
otherwise the coarser parent wins and its subtree is suppressed; ties go to the parent. Exclude the root;
uncovered points are noise. This is excess-of-mass (EOM) selection, and it is precisely the thing DBSCAN
could not do: a diffuse cluster and a dense cluster can *both* appear in the final labeling because they
are selected at different levels of the same tree. That is the direct cure for the multi-density blobs a
single `eps` mishandled — the tight knot lives at high `lambda`, the loose cloud at low `lambda`, and the
DP takes each at its own level instead of forcing one horizontal cut to choose between them (which is
exactly what produced DBSCAN's 0.59–0.76 seed wobble, one blob or another falling on the wrong side of
the one radius).

Now the edit, and the scaffold decides how literal I get. `sklearn.cluster.HDBSCAN` realizes all of the
above — mutual-reachability single-linkage, size-based condensation, EOM extraction — so I do not
hand-roll the MST and the tree DP; I instantiate it. The two knobs that matter are `min_cluster_size`
(the `m_clSize` granularity computed above) and `min_samples` (the `m_pts` density smoothing). I set
`min_cluster_size = max(5, n // 50)` and `min_samples = 5`, a small fixed smoothing, with
`cluster_selection_method = "eom"` for the excess-of-mass extraction I just derived. HDBSCAN determines
its own cluster count, so it ignores the harness's `n_clusters` — which is the whole point, the
self-determination DBSCAN had and K-Means lacked. But there is one harness-specific guard the abstract
method does not carry, and I keep it because the high-dimensional setting demands it: if HDBSCAN labels
*everything* noise — `len(set(labels)) ≤ 1`, the degenerate case the silhouette scores at −1.0 — I fall
back to `KMeans(k)` with the harness's `n_clusters`, so even if density estimation gives up in 64-D I
return `k` honest clusters instead of the collapse that killed DBSCAN on digits. That fallback is
exactly the previous method's lesson folded in as a safety net: never hand the harness a degenerate labeling.
And I should be clear-eyed that the guard is not decoration on digits — the mutual-reachability MST is
still built on 64-D distances, and those distances still concentrate into a thin shell exactly as they
did for DBSCAN's `eps`. The difference is that HDBSCAN never commits to *one* level: even a concentrated
metric still has a most-stable few components, so I expect it to recover the cleanest digit clumps and
dump the rest to noise rather than collapse outright — but the concentration means the density signal on
digits is weak, so the KMeans fallback is real insurance against the tail where even the stablest split
is below `m_clSize`. The full class is in the answer.

Reading K-Means's and DBSCAN's numbers together, here is what I expect. On **moons**, HDBSCAN should
recover DBSCAN's win and likely exceed it — the self-tuning density model finds the two arcs without a
hand-set `eps`, so ARI should be at or near 1.0, far above K-Means's 0.481; but by the silhouette
argument above, that gain shows up in ARI/NMI while the moons silhouette stays modest near DBSCAN's 0.22,
so I will judge moons by ARI/NMI. On **blobs**, it should beat DBSCAN's capped 0.70: selecting clusters
at *different* density levels directly cures the varying-`cluster_std` problem a single `eps` could not,
so I expect ARI in the high-0.7s to low-0.9s, competitive with K-Means. On **digits**, the honest
expectation is *non-degenerate but modest*: density estimation in 64-D is genuinely hard, so HDBSCAN will
likely leave many points as noise and recover only the cleanest digit clusters — ARI well above DBSCAN's
0.0003 but probably below K-Means's 0.534, with the KMeans fallback as insurance against outright
collapse. The bet: this should be the only method so far that is non-degenerate on *all three* geometries
at once, so even where it trails K-Means on a single setting its geometric mean tops both — a method whose
*weakest* factor is a real positive number beats one pinned by a 0.45 moons factor and one that is
degenerate. If 64-D density estimation drags digits low enough to sink the gmean below K-Means's, the
residual left standing is exactly that setting, where the better move would pair a density hierarchy's
shape sense with a dimension-robust embedding — but within this edit surface, the self-tuning multi-level
density model is the strongest single algorithm I can land.