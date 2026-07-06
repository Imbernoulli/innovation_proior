The default fill in the scaffold is a plain K-Means: it hands the harness *some* labels so nothing
crashes, but it is the floor, not an answer — and I can name its disease before I run a thing. K-Means
sends every point to its nearest of `k` centroids, and "nearest centroid" is the definition of a
Voronoi diagram, whose cells are convex. So every cluster K-Means can produce is convex. Two of the
three geometries on this task are not convex problems: moons is two interleaving half-circles that no
pair of half-planes can separate, and digits is a 64-dimensional embedding whose class manifolds are
nothing like isotropic balls. K-Means will do fine on blobs (that is exactly its model) and badly the
moment the structure stops being round. It is also told `k` and forces every point into a cluster — no
notion of noise. I want the first real rung to attack the convexity wall head-on, but before I reach
for any one tool I should walk the shelf of baselines the context actually hands me and let the choice
of *density* be a conclusion I am forced into, not a reflex.

Take the three families in turn. The partitioning family — k-means, k-medoids/PAM, CLARANS — all pick
`k` representatives and assign each point to the nearest, which is the very nearest-representative rule
that makes convex Voronoi cells; PAM swaps the mean for an actual data point as medoid and CLARANS just
samples the swap search, but the assignment geometry is unchanged. Every one of them carries the same
three diseases as the default fill: convex cells, a mandatory `k`, and no concept of noise. They cannot
be the answer to a task that puts moons and a 64-D manifold in front of me. The hierarchical/connectivity
family is more interesting, because single-linkage agglomerative clustering needs no `k` up front and,
by merging nearest points, can in principle thread along a non-convex arc — that is the one property
K-Means structurally lacks, so it tempts me. But single-linkage has a notorious failure mode that this
exact task will trigger: it chains through *any* thin bridge of points, so the harness's `noise = 0.08`
jitter sprinkled between the two moons, or a single stray filament, will fuse the two arcs into one
component well before the dendrogram cut can separate them. And it forces every point into the tree —
there is no explicit "this point belongs to nothing," so I would have to invent a cut threshold, and by
the time I cut, the bad merge has already happened lower in the tree. Complete- or Ward-linkage cures
the chaining but does so precisely by preferring compact, near-spherical merges, which reimposes the
convex bias I am trying to escape. Cost is also real: the connectivity variant is `O(n²)` in time and
memory, which for `n` up to 1797 is a few million entries — affordable, but not free. The grid/histogram
density family bins the space, calls high-count cells cores, and lays boundaries in the histogram valleys,
which genuinely allows arbitrary shapes in the plane. But grids die on dimension by simple counting: with
`b` bins per axis the number of cells is `b^d`, and at `d = 64` even the coarsest possible `b = 2` gives
`2^64 ≈ 1.8·10^19` cells for 1797 points, so every single point sits alone in its own cell and no cell
ever clears a count threshold of two. The curse of dimensionality erases grid density outright on digits
— and I note that same arithmetic as a warning shot fired at *any* neighbor-counting method in 64-D,
mine included, before I have even chosen it.

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

Let me trace this once on the moons picture in my head to be sure the machinery actually threads a
non-convex arc and actually stops at the noise bridge, rather than asserting it does. Walk along one
half-circle: consecutive points on a dense arc are a nearest-neighbor spacing apart, well under `eps`,
and each has a full `eps`-ball of arc on both sides, so each is core and each is directly
density-reachable from its neighbor — the chain steps point-to-point all the way around the curve, and
at no step did I need the arc to be straight, so the whole crescent comes out as one density-connected
component even though a straight bisector could never enclose it. Now the failure I *want*: the `0.08`
jitter drops a handful of stray points into the gap between the two moons. A bridge point there has an
`eps`-ball that pokes into near-empty space between the arcs, so its neighbor count sits far below
`min_samples = 10` — it is a border point at best, never core. Density-reachability requires every link
except the final endpoint to be core, so a chain from arc A cannot pass *through* the bridge point to
reach arc B: the bridge can be reached and absorbed as noise-adjacent border, but it can never extend the
component across the gap. That is the precise repair of single-linkage's disease — single-linkage would
have merged the arcs across that same bridge because it never asked the bridge to be dense. The core
requirement is doing exactly the work I recruited it to do.

That is the method — density-based clustering, no `k` required (the number of clusters falls out of how
many connected dense pieces exist), arbitrary shape, explicit noise. Now the edit. The scaffold lets me
fill `CustomClustering`, and I have `scikit-learn` in scope, so I do *not* reimplement the flood-fill
over core points by hand here — `sklearn.cluster.DBSCAN` already realizes exactly this construction
(core test by neighbor count, connected components of core points with border points absorbed, noise
labeled `-1`). Its cost is a neighbor query per point: `O(n log n)` with a spatial index in low
dimensions, degrading toward the dense `O(n²)` as dimension climbs and the index stops helping — at
`n ≤ 1797` that worst case is about 3.2 million pairwise checks, entirely affordable, so runtime is a
non-issue and I can spend my attention where it belongs. The rung's real content is therefore the two
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
weakness lives. For the low-D settings I can read it off the data's scale, and I have a concrete
calibration anchor: on StandardScaled 2-D the demo `eps = 0.3` works for tight blobs
(`cluster_std ≈ 0.4`), but *this* task's blobs run much looser (`cluster_std` up to 1.5). Let me push
the arithmetic instead of trusting the anchor blindly. StandardScaling rescales each axis so the whole
1500-point cloud has unit variance per axis; with five reasonably separated blobs, most of that unit
variance is the *between-centroid* spread, which pins the typical center-to-center spacing at order one
in standardized units. Each blob holds about `1500 / 5 = 300` points. If the loosest blob has
standardized radius `r`, its nearest-neighbor spacing scales like `r / √300 ≈ r / 17`, comfortably under
`0.22` for any `r` that still looks like a blob — so at `eps = 0.22` a chain of within-blob steps
connects the interior with room to spare. The failure at the demo's `0.3` is the *other* direction:
`0.3` is large enough that the rim of one loose blob reaches across the gap into a neighbor's rim,
fusing two true clusters, whereas dropping to `0.22` pulls the reach back inside the gap. So for
`n_features ≤ 3` I set `eps = 0.22`, `min_samples = 10` — a single fixed pair that the StandardScaler
makes meaningful across blobs and moons alike. I am aware this is a compromise, and I can even name the
shape of it: one global `eps` cannot be simultaneously smaller than every within-blob spacing (so the
dense blob stays connected) and smaller than every between-blob gap (so loose blobs stay apart), when
`cluster_std` ranges from 0.5 to 1.5 and the two conditions pull in opposite directions. The varying
`cluster_std` on blobs is exactly the case a single radius handles least gracefully — that is the
structural cost of a global threshold, and I accept it for the first rung because the alternative,
re-deriving a per-region radius, is a later method's job, not this one's.

The high-dimensional setting (digits, 64 features) is where I expect the most trouble and where I refuse
to hard-code a number, so I derive `eps` from the data via the classical `k`-distance knee. For each
point compute the distance to its `k`-th nearest neighbor and sort those values descending: points deep
in a dense region have a small `k`-distance, points out in the sparse void have a large one, so the
sorted curve starts high (noise) and falls to low (cluster), with a *knee* at the transition. The
`k`-distance at that knee is the right `eps` — it is the density of the thinnest region I am still
willing to call a cluster. I detect the knee Kneedle-style: treat the sorted curve as points, draw the
chord from its first to its last point, and take the curve point of maximum perpendicular distance from
that chord; its `k`-distance is `eps`. Before trusting the detector on the hard case, let me watch it
fire on an easy one where a scale genuinely exists. Suppose the data is a clean dense cloud plus a scatter
of far outliers: for the cloud points the `k`-distance is some small value `a`, for the outliers it is a
large value `b ≫ a`, and the sorted-descending curve is then flat-high at `b` over the outlier fraction,
drops sharply, and flat-low at `a` over the rest — an L lying on its back. The chord from the first
(high, left) point to the last (low, right) point runs diagonally across the L, and the curve point
sitting farthest below that diagonal is exactly the corner of the L, whose `k`-distance is the value at
the transition — neither the outlier scale `b` nor the tight-interior scale `a`, but the threshold that
separates them. That is the `eps` I want, and the detector recovers it cleanly *because the curve has a
corner*. The whole method leans on that corner existing. Concretely, with sorted values `y` over integer
indices `x`, the
perpendicular distance from index `i` to the chord through the endpoints is
`|(y2−y1)·x_i − (x2−x1)·y_i + x2·y1 − y2·x1| / √((x2−x1)² + (y2−y1)²)`. It is worth a unit check before I
trust it: the numerator is twice the signed area of the triangle formed by the endpoints and point `i`
(units of `x`·`y`), and the denominator is the chord length (units of `√(x²+y²)`), so the quotient has
the units of a perpendicular offset in the `y`-scaled plane, which is what I want to be maximizing. The
degenerate case is telling too: if the sorted curve were a perfectly straight line, every point lies on
the chord, the numerator is zero everywhere, and the `argmax` is meaningless — which is precisely the
"there is no knee, hence no natural scale" situation, and I should remember that a flat curve is not a
bug in the detector but a message about the data. For the high-D bar I scale `min_samples` with
dimension, `max(4, min(2·dim, 10))`; at `dim = 64` the `2·dim = 128` is capped by the `10`, so
`min_samples = 10` and the matching `k = min(10, n−1) = 10` — the cap binds and I land back at the same
smoothing count as low-D, which is fine. I keep one convention straight: the `k`-th nearest neighbor
excludes the point itself, while the core test counts the self-point, so I let `NearestNeighbors` return
`k+1` and read the last column.

So the step-1 edit is: density-based clustering via `sklearn.cluster.DBSCAN`, with `eps = 0.22`,
`min_samples = 10` for `n_features ≤ 3`, and a `k`-distance-knee `eps` with dimension-scaled
`min_samples` for high-D; `predict` returns the fitted labels. The whole class is in the answer.

Now reason about what this floor should do, because that is the point of running it. On **moons**,
DBSCAN should shine — two dense interleaving arcs are exactly the non-convex, follow-the-backbone case,
and at a well-chosen low `eps` the two half-moons are separate dense components with the noise jitter
peeled off, so ARI/NMI should be high (far above K-Means's convex split, which slices each moon in
half). On **blobs**, it should be respectable but *not* dominant: the varying `cluster_std` is precisely
the global-`eps` weakness — at `eps = 0.22` the tightest blob is well-resolved while the loosest one
sheds its fringe to noise or fuses with a neighbor, so ARI will be solid but visibly under the convex
methods that blobs reward, and because the shed fringe carries the `-1` noise label the silhouette term
will be dragged down even where ARI holds; I also expect a real seed-to-seed spread, since which blob
happens to fall on the wrong side of the single radius shifts with the draw. **Digits** is the one I am
genuinely worried about, and the grid arithmetic already told me why in miniature. In 64 dimensions
distances concentrate: if I model the coordinates as roughly independent, then `||x − y||²` is a sum of
about 64 comparable per-axis terms, so its mean grows like `d` while its standard deviation grows only
like `√d`, and the coefficient of variation of a pairwise distance shrinks like `1/√d ≈ 1/8`. Every
pairwise distance is squeezed into a thin shell about a tenth wide, which means the *sorted* `k`-distance
curve is nearly flat — and a nearly flat curve is exactly the degenerate no-knee case I flagged in the
unit check. The most likely failure is that the chord-perpendicular detector latches onto noise on the
flat curve, the derived `eps` comes out too small, almost everything is declared noise, and the labeling
collapses toward a single residual non-noise cluster — which the harness scores `silhouette = −1.0` and
near-zero ARI/NMI. If that happens, it is not a bug in DBSCAN; it is the diagnosis that *global density
thresholding does not survive the curse of dimensionality*, and it points straight at the next rung. So
my falsifiable expectation entering the run is: moons strong, blobs solid-but-capped with a live seed
spread and a silhouette pulled down by noise labels, and digits at serious risk of a degenerate
collapse to `silhouette = −1.0` — and whichever way digits lands is the number the next method has to
answer for.
