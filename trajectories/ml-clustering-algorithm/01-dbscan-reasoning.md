The default fill in the scaffold is a plain K-Means: it hands the harness *some* labels so nothing
crashes, but it is the floor, not an answer — and I can name its disease before I run a thing. K-Means
sends every point to its nearest of `k` centroids, and "nearest centroid" is the definition of a
Voronoi diagram, whose cells are convex. So every cluster K-Means can produce is convex. Two of the
three geometries on this task are not convex problems: moons is two interleaving half-circles that no
pair of half-planes can separate, and digits is a 64-dimensional embedding whose class manifolds are
nothing like isotropic balls. K-Means will do fine on blobs (that is exactly its model) and badly the
moment the structure stops being round. It is also told `k` and forces every point into a cluster — no
notion of noise. I want my first real attempt to attack the convexity wall head-on, and the shelf of
baselines the context hands me narrows the choice fast.

The partitioning family — k-means, k-medoids/PAM, CLARANS — all assign each point to a nearest
representative, so they share the default fill's three diseases: convex cells, a mandatory `k`, no noise.
The hierarchical/connectivity family tempts me, because single-linkage agglomerative clustering needs no
`k` and, by merging nearest points, can thread along a non-convex arc — the one property K-Means
structurally lacks. But single-linkage chains through *any* thin bridge: the moons' `noise = 0.08` jitter
sprinkled between the two arcs will fuse them into one component well before any dendrogram cut can
separate them, and there is no explicit "belongs to nothing," so the bad merge happens lower in the tree
than wherever I cut. Complete/Ward-linkage cures the chaining only by preferring compact, near-spherical
merges — the convex bias again. The grid/histogram family allows arbitrary shapes in the plane but dies
on dimension by counting: with `b` bins per axis there are `b^d` cells, and at `d = 64` even `b = 2`
gives `2^64 ≈ 1.8·10^19` cells for 1797 points, so every point sits alone and no cell ever clears a count
of two. That same arithmetic is a warning shot at *any* neighbor-counting method in 64-D, mine included.

So the shelf pushes me to exactly one place: I want the shape-freedom that only the connectivity idea
gives, but with single-linkage's fragility (chaining, no noise) repaired, and without a grid that dies on
dimension. The repair is to require *density* before I am allowed to connect two points — to walk in
small steps, but only ever through regions that are genuinely dense, so a thin noise bridge is not dense
enough to be walked across. That is the one fact my eye actually uses when it picks groups out of a
scatter: the groups are *dense* and the gaps and outliers are *sparse*. Density is the signal. Let me
build the simplest honest formalization of exactly that, and nothing fancier — density measured by
neighborhoods rather than bins, so I inherit none of the grid's cell-count blowup.

The crudest measure of "how dense is it around a point `p`" is a count: how many points sit within some
radius. Fix a radius `eps` and define the `eps`-neighborhood `N_eps(p) = { q : dist(p,q) ≤ eps }`; the
local density at `p` is `|N_eps(p)|`. This is metric-agnostic — with Euclidean `dist` the neighborhood
is a ball — and it is the only primitive I will need. Here the harness has already StandardScaled the
data, so `eps` is measured in standard-deviation units and one radius means roughly the same thing
across the feature axes; that scaling is what makes a *fixed* `eps` even thinkable.

First instinct: a cluster is a region where every point has high density — every cluster point `p` has
`|N_eps(p)| ≥ MinPts`. Test it on a picture and it breaks immediately, and I can make the break
quantitative instead of impressionistic. Model a cluster locally as a uniform sheet of points with area
density `ρ`. A point deep in the interior has its whole `eps`-ball inside the sheet, so it sees about
`ρ · π · eps²` neighbors. A point sitting on a straight stretch of the cluster's rim has half its
`eps`-ball poking out into the empty void, so it sees about `ρ · π · eps² / 2` — a clean factor of two
fewer — and a point at a convex tip of the boundary, where the ball pokes out over more than a
half-plane, sees fewer still. So the interior and the rim of one genuine cluster differ in neighbor
count by a factor of roughly two purely because of *where in the cluster* they sit. Now the vise: if I
set `MinPts` high enough that it means "this is dense" (calibrated to the interior count `ρπeps²`), I
reject essentially the entire rim, gnawing every cluster down to its core and discarding its edges; if I
lower `MinPts` to about `ρπeps²/2` so the rims qualify, the bar is now so low that a loose clump of
noise at half the interior density also clears it. No single count threshold includes borders and
excludes noise, because border density is an artifact of being at the edge, not a property of the
cluster. That factor-of-two rim deficit is the wall the whole construction has to climb.

The escape is to stop demanding high density of *every* cluster point and demand it of every cluster
point's *vicinity*. A border point may be sparse itself, provided it lies within `eps` of a genuinely
dense point that vouches for it. Split the points: call `p` **core** if it clears the bar itself,
`|N_eps(p)| ≥ MinPts` — the dense interior — and call a point that fails the bar but lies within `eps`
of a core point a **border** point, riding on a core neighbor's density. Make this one-directional and
precise: `p` is *directly density-reachable* from `q` if `p ∈ N_eps(q)` and `q` is core. The relation
is symmetric between two core points but asymmetric at the core/border seam — a core `q` reaches its
border neighbor `p`, but the border `p`, failing the core test, cannot vouch for anyone. That asymmetry
is the formal statement that density flows outward from the dense interior to the sparse rim, and it is
exactly the guard single-linkage was missing: a sparse bridge point is a border point at best, never a
core, so it can be *reached* but can never *extend* a cluster across itself.

One step of "directly density-reachable" reaches one ring out. To trace a whole arbitrary-shaped region
I chain it — exactly the "walk in small steps" connectivity idea, made concrete as a walk through core
points: `p` is *density-reachable* from `q` if there is a chain `q = p_1, p_2, …, p_n = p` with each
`p_{i+1}` directly density-reachable from `p_i`. Every link except possibly the last endpoint must be
core. This is transitive by construction, and shape-free: at no point did I ask the region to be a ball
or a Voronoi cell — I only ever asked the next point to be a dense neighbor of the current one. So I can
thread along a curved half-moon or around one cluster wrapped in another. Arbitrary shape falls out of
"follow the dense backbone" — which is precisely the property K-Means cannot have, and precisely the
property single-linkage had but ruined by connecting through non-dense bridges. Two border points on
opposite rims of the same cluster are each reachable from an interior core point but not from each
other, so to certify membership I route through that common witness: `p` is *density-connected* to `r`
if some core `o` has both density-reachable from it. That relation is symmetric, and a **cluster**
becomes a maximal, density-connected set; **noise** is the cheap leftover, everything that belongs to no
cluster. Noise is not a special case to detect — it is what is not dense-connected to anything, which
hands me the explicit-noise property K-Means lacks and moons will need (the harness's 0.08 jitter
points should fall out as noise rather than corrupt a half-moon).

That is the method — density-based clustering, no `k` required (the number of clusters falls out of how
many connected dense pieces exist), arbitrary shape, explicit noise. Now the edit. The scaffold lets me
fill `CustomClustering`, and I have `scikit-learn` in scope, so I do *not* reimplement the flood-fill
over core points by hand here — `sklearn.cluster.DBSCAN` already realizes exactly this construction
(core test by neighbor count, connected components of core points with border points absorbed, noise
labeled `-1`). Its cost is a neighbor query per point: `O(n log n)` with a spatial index in low
dimensions, degrading toward the dense `O(n²)` as dimension climbs and the index stops helping — at
`n ≤ 1797` that worst case is about 3.2 million pairwise checks, entirely affordable, so runtime is a
non-issue and I can spend my attention where it belongs. The real content is therefore the two
parameters DBSCAN exposes, `eps` and `min_samples`, because the abstract method is only as good as the
radius I pick, and a global `eps` is the method's one genuine weakness.

Take `min_samples` first — the density-smoothing count, the bar for "dense enough to be core." Its job
is to stabilize the density estimate: too small and a couple of stray points clumping look like a
cluster; large enough and the estimate is steady. The well-known empirical fact is that the result is
*insensitive* to it over a reasonable range, so I can essentially fix it. For low-dimensional
StandardScaled data the standard choice is a small constant on the order of ten — the sklearn DBSCAN
demonstration uses `min_samples = 10` on StandardScaled 2-D blobs — so for the low-D settings (blobs,
moons; ≤3 features) I pin `min_samples = 10`.

`eps` is the parameter that actually decides the clustering, and it is where the global-threshold
weakness lives. The demo `eps = 0.3` works for tight StandardScaled 2-D blobs (`cluster_std ≈ 0.4`), but
this task's blobs run to `cluster_std = 1.5`, and the arithmetic says tighten: each blob holds
`1500 / 5 = 300` points, so if the loosest has standardized radius `r` its nearest-neighbor spacing is
about `r / √300 ≈ r / 17`, well under `0.22` — a chain of within-blob steps connects the interior with
room to spare. The demo's `0.3` fails the *other* way: it is large enough that the rim of one loose blob
reaches across the gap into a neighbor, fusing two true clusters; `0.22` pulls the reach back inside the
gap. So for `n_features ≤ 3` I set `eps = 0.22`, `min_samples = 10`. This is a compromise with a
nameable shape: one global `eps` cannot be smaller than every within-blob spacing *and* smaller than
every between-blob gap when `cluster_std` ranges 0.5–1.5 — the two conditions pull opposite ways. That is
the structural cost of a global threshold, and I accept it here because re-deriving a per-region radius
is a later method's job.

The high-dimensional setting (digits, 64 features) is where I expect the most trouble and where I refuse
to hard-code a number, so I derive `eps` from the data via the classical `k`-distance knee. For each
point compute the distance to its `k`-th nearest neighbor and sort those descending: points deep in a
dense region have a small `k`-distance, points out in the sparse void a large one, so the sorted curve
starts high (noise) and falls to low (cluster), with a *knee* at the transition. The `k`-distance at that
knee is the right `eps` — the density of the thinnest region I still call a cluster. I detect it
Kneedle-style: draw the chord from the curve's first to its last point and take the point of maximum
perpendicular distance from that chord. On a clean case — a dense cloud (`k`-distance `a`) plus far
outliers (`k`-distance `b ≫ a`) — the sorted curve is an L lying on its back, flat-high at `b`, a sharp
drop, flat-low at `a`; the chord runs diagonally across the L and the farthest point below it is the
corner, whose value is the threshold separating the two scales. That is the `eps` I want, and it exists
only *because the curve has a corner*. The degenerate case is the tell: a perfectly straight sorted curve
puts every point on the chord, the perpendicular distances are all zero, and the argmax is meaningless —
precisely "there is no natural density scale," a message about the data, not a bug. For the high-D bar I
scale `min_samples = max(4, min(2·dim, 10))`; at `dim = 64` the `128` is capped by the `10`, so
`min_samples = 10` and the matching `k = min(10, n−1) = 10` — I land back at the same smoothing count as
low-D. One convention: the `k`-th nearest neighbor excludes the point itself while the core test counts
it, so I let `NearestNeighbors` return `k+1` and read the last column.

So the edit is: density-based clustering via `sklearn.cluster.DBSCAN`, with `eps = 0.22`,
`min_samples = 10` for `n_features ≤ 3`, and a `k`-distance-knee `eps` with dimension-scaled
`min_samples` for high-D; `predict` returns the fitted labels. The whole class is in the answer.

Now what this floor should do. On **moons**, DBSCAN should shine — two dense interleaving arcs are
exactly the follow-the-backbone case, and at a low `eps` the arcs come out as separate dense components
with the jitter peeled to noise, so ARI/NMI should be high, far above K-Means's convex half-split. On
**blobs**, respectable but not dominant: at `eps = 0.22` the tightest blob resolves while the loosest
sheds its fringe to noise or fuses with a neighbor, so ARI is solid but under the convex methods, the
`-1` noise labels drag silhouette even where ARI holds, and I expect a real seed-to-seed spread since
which blob falls on the wrong side of the single radius shifts with the draw. **Digits** is the one I am
genuinely worried about. In 64 dimensions distances concentrate: modeling coordinates as roughly
independent, `||x − y||²` is a sum of ~64 comparable terms, so its mean grows like `d` but its standard
deviation only like `√d`, and the coefficient of variation shrinks like `1/√d ≈ 1/8` — every pairwise
distance squeezed into a thin shell. That is exactly the flat, no-knee curve I flagged: the
chord-perpendicular detector latches onto noise, the derived `eps` comes out too small, almost everything
is declared noise, and the labeling collapses toward one residual cluster — which the harness scores
`silhouette = −1.0`. If that happens it is not a DBSCAN bug; it is the diagnosis that *global density
thresholding does not survive the curse of dimensionality*, and it points straight at what the next
method must fix.
