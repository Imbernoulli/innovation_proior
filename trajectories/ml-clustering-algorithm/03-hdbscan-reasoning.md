K-Means did what I traded for and showed me its own ceiling in the same numbers. On digits it rescued
the dead setting decisively — ARI 0.534, NMI 0.671, silhouette 0.139, all three off DBSCAN's floor
(0.0003 / 0.011 / −1.0) — because ten honest centroids in 64-D cannot collapse the way a global `eps`
did. On blobs it beat DBSCAN clearly, ARI 0.853 mean against 0.70, vindicating "convex isotropic
Gaussians are K-Means's exact model." But moons is where the cost landed exactly as I called it:
ARI 0.481, NMI 0.383 — barely better than a coin flip on a two-class problem — against DBSCAN's 0.972.
Nearest-centroid is a Voronoi tessellation; it *must* cut each half-moon with a straight bisector, so it
sliced the interleaving arcs the wrong way.

Let me put both rungs' tables side by side on the actual scoring rule before I decide anything, because
the aggregate is a geometric mean and I want the ladder's shape in numbers, not impressions. K-Means's
three setting-means are blobs `(0.853 + 0.874 + 0.585)/3 ≈ 0.771`, moons `(0.481 + 0.383 + 0.494)/3 ≈
0.453`, and digits `(0.534 + 0.671 + 0.139)/3 ≈ 0.448`, whose geometric mean is `(0.771 · 0.453 ·
0.448)^{1/3} = (0.1565)^{1/3} ≈ 0.54`. That 0.54 is finite and real — nothing degenerate — and it is the
number I now have to beat. DBSCAN's aggregate, by contrast, had a digits setting-mean of about `−0.33`
(the −1.0 silhouette dragging it negative), which makes its geometric mean ill-defined — degenerate —
so on the scoring rule DBSCAN is *below* K-Means despite owning moons outright. So the ladder's shape is
crisp: DBSCAN owns non-convex moons and dies on high-D digits; K-Means owns convex blobs and digits and
dies on non-convex moons. Each is excellent-on-some, terrible-on-another, and on a geometric-mean
aggregate that is exactly the wrong profile — the mean is pulled toward its *weakest* factor, so both
rungs are held down, K-Means by its moons 0.453 and (would-be) DBSCAN by its digits −0.33.

There is a second signal in the tables I should not miss, because it will shape what "winning moons" is
even allowed to look like. Compare the two moons rows on silhouette: DBSCAN, with a near-perfect ARI of
0.972, scored silhouette only 0.224, while K-Means, with a *wrong* ARI of 0.481, scored silhouette
0.494 — higher. The intrinsic metric actively *prefers* the incorrect convex split, because two
interleaving crescents are close in Euclidean space and the compact left/right cut looks more
"separated" to silhouette than the true arcs do. So silhouette and ARI point in opposite directions on
moons, and I should file the consequence now: whatever method recovers the correct arcs will show its
moons gain in ARI and NMI, and its moons silhouette will stay low — near DBSCAN's 0.22, not up at
K-Means's 0.49. I will not read a modest moons silhouette as a failure later; it is baked into the
geometry.

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

Now the options. I could keep DBSCAN and just *search* `eps` — run it at a grid of radii and pick the
best — but "best by what?" I have no labels at fit time, and every single run is still one horizontal
density level, so a grid of horizontal cuts still cannot place the knot's cluster and the cloud's
cluster in the *same* output; the search cannot buy me what one level structurally lacks. I could go the
K-Means direction again — a fixed-`k` partition is guaranteed non-degenerate — but that is the rung I
just built, and it throws moons away; I am here precisely to *not* do that. Or I make the radius *local*,
letting each region declare its own density scale, which is the direction that could actually work.
OPTICS is the classical version of that idea — it produces a reachability ordering with a per-point
reachability distance instead of a single `eps` — but extracting a flat clustering from an OPTICS
reachability plot needs a steepness threshold `ξ`, which is a new global knob heuristically chosen, and
its reachability is *asymmetric* (defined one-sidedly from the ordering), which is what forces that
heuristic. So the local-radius idea is right but OPTICS's particular realization just relocates the knob.
Let me take the local-radius idea and remove the knob by construction.

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
`eps ≥ max(d_core(p), d_core(q), d(p,q))`. Check it on a concrete pair with `m_pts = 5`: let
`d_core(p) = 0.2` (p sits in a moderately dense spot, its 5th neighbor at 0.2), `d_core(q) = 0.35`, and
the raw gap `d(p,q) = 0.15`. The three conditions "p is core," "q is core," and "p, q within eps" first
hold simultaneously at `eps = max(0.2, 0.35, 0.15) = 0.35` — below that, at say `eps = 0.25`, p is core
and the pair is within eps but q has not yet reached core, so they are *not* yet linked, exactly as the
max predicts. The binding constraint is q's sparsity, and the mutual-reachability distance reports it as
0.35 rather than the deceptively small raw 0.15. Define that maximum as the **mutual reachability distance**
`d_mreach(p,q) = max(d_core(p), d_core(q), d(p,q))`. It is symmetric, reduces to ordinary distance when
both core distances are small, and *inflates* a pair when either endpoint is in a sparse region — a
sparse point cannot be closer than its own core distance to anything. That inflation is exactly the
defense against single-linkage chaining, and it is worth a concrete number to be sure it does the work.
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
digits and too fine on the 1000-point moons, so I scale it: `m_clSize = max(5, n // 50)`. That reads out
to `1797 // 50 = 35` on digits, `1500 // 50 = 30` on blobs, and `1000 // 50 = 20` on moons — a check
confirms these are all about `35/1797 ≈ 1.9%`, `30/1500 = 2.0%`, `20/1000 = 2.0%` of their sets, i.e. a
consistent ~2% floor on what counts as a cluster, which is the invariance I wanted across the three
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
dynamic program: `S_hat(C_i) = S(C_i)` at a leaf, else `max(S(C_i), Σ_children S_hat)`. Trace the rule on
a single node to be sure it does what I mean: if a parent has two children whose propagated stabilities
sum to more than the parent's own, `S_hat(parent) = Σ_children` and the parent is deselected, so the
finer split is kept; if the children sum to less, the parent's own stability wins, `S_hat(parent) =
S(parent)`, and its whole subtree is suppressed in favor of the single coarser cluster. Ties go to the
parent. Exclude the root; uncovered points are noise. This is excess-of-mass (EOM) selection, and it is
precisely the thing DBSCAN could not do: a diffuse cluster and a dense cluster can *both* appear in the
final labeling because they are selected at different levels of the same tree — the knot from high
`lambda`, the cloud from low `lambda`, in one output.

Let me run the multi-density blobs through this once with invented-but-consistent numbers, because that
setting is the exact one a single `eps` mishandled and I want to see the machinery cure it. Say the tight
blob (`cluster_std ≈ 0.5`) is a dense knot whose points stay together from a low birth level up to a high
`lambda` before they finally scatter, giving it a wide membership interval and a stability of, say,
`S(knot) = 12`; the loose blob (`cluster_std ≈ 1.5`) is a diffuse cloud that exists only over a narrow
band of low `lambda` — it never reaches high density and dissolves early — with a smaller but real
`S(cloud) = 5`. A single horizontal cut is forced to choose one level for both: cut high to resolve the
knot and the cloud's points are all below that density, so the cloud vanishes into noise; cut low to keep
the cloud and the knot merges with whatever neighbor shares that loose level. Either horizontal choice
loses one blob — which is precisely DBSCAN's 0.59–0.76 seed wobble, one blob or another falling on the
wrong side of the one radius. Now the EOM DP: suppose the condensed tree has a parent node (the merged
knot-plus-cloud region, born low) whose own stability as a single cluster is `S(parent) = 9`, and whose
two children are the knot (`S_hat = 12`) and the cloud (`S_hat = 5`). The rule compares `S(parent) = 9`
against `Σ_children S_hat = 12 + 5 = 17`; since `17 > 9`, the parent is deselected and both children are
selected, so the knot and the cloud *both* appear in the flat labeling, each chosen at its own level. Had
the cloud been genuinely spurious — say `S_hat = 1` so the child sum `13` still beats `9`, or a fringe
piece below `m_clSize` that never became a node at all — the DP would keep the split only where a real
second cluster exists, and otherwise let the parent stand. That is the horizontal cut's failure repaired
by construction: nothing forces the knot and the cloud to be read at the same density.

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
exactly the lesson of step 2 folded in as a safety net: never hand the harness a degenerate labeling.
And I should be clear-eyed that the guard is not decoration on digits — the mutual-reachability MST is
still built on 64-D distances, and those distances still concentrate into a thin shell exactly as they
did for DBSCAN's `eps`. The difference is that HDBSCAN never commits to *one* level: even a concentrated
metric still has a most-stable few components, so I expect it to recover the cleanest digit clumps and
dump the rest to noise rather than collapse outright — but the concentration means the density signal on
digits is weak, so the KMeans fallback is real insurance against the tail where even the stablest split
is below `m_clSize`. The full class is in the answer.

Reading K-Means's and DBSCAN's numbers together, here is what I expect and why it should be the
strongest rung. On **moons**, HDBSCAN should recover DBSCAN's win and likely exceed it — the
hierarchical, self-tuning density model finds the two arcs without a hand-set `eps`, so ARI should be at
or near 1.0, far above K-Means's 0.481 and at least matching DBSCAN's 0.972; but by the silhouette
argument above, that gain shows up in ARI and NMI, and the moons silhouette will stay modest near
DBSCAN's 0.22, not near K-Means's 0.49, so I will judge moons by ARI/NMI. On **blobs**, it should
beat DBSCAN's capped 0.70: selecting clusters at *different* density levels is the direct cure for the
varying-`cluster_std` problem that a single `eps` could not handle, so I expect ARI up around the
high-0.7s/low-0.9s, competitive with K-Means and with less of DBSCAN's 0.17 seed spread. On **digits**,
the honest expectation is *non-degenerate but modest*: density estimation in 64-D is genuinely hard, so
HDBSCAN will likely leave many points as noise and recover only the cleanest digit clusters — I expect
ARI well above DBSCAN's 0.0003 but probably below K-Means's 0.534, with the KMeans fallback as insurance
against an outright collapse. The falsifiable claim against the prior rungs: HDBSCAN should be the only
method on the ladder that is non-degenerate on *all three* geometries at once — moons near-perfect,
blobs strong, digits real — so even where it trails K-Means on a single setting, its geometric-mean
aggregate should top both. That is the arithmetic that matters: K-Means's aggregate was pinned near 0.54
by its dead-flat moons factor of 0.45, and DBSCAN's was degenerate; a method whose *weakest* setting is
a real positive number instead of 0.45 or −0.33 will have a geometric mean above both even if no single
setting is a record. If digits comes in low enough to drag the gmean below K-Means's, that is the
residual the ladder ends on: the high-dimensional setting where neither density estimation nor a fixed-`k`
partition is clearly right, and where the genuinely better move would combine a density-hierarchy's shape
sense with a dimension-robust embedding — but within this harness's edit surface, the self-tuning
multi-level density model is the strongest single algorithm I can land.
