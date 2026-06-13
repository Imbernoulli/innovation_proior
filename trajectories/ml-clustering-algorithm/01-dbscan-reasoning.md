The default fill in the scaffold is a plain K-Means: it hands the harness *some* labels so nothing
crashes, but it is the floor, not an answer — and I can name its disease before I run a thing. K-Means
sends every point to its nearest of `k` centroids, and "nearest centroid" is the definition of a
Voronoi diagram, whose cells are convex. So every cluster K-Means can produce is convex. Two of the
three geometries on this task are not convex problems: moons is two interleaving half-circles that no
pair of half-planes can separate, and digits is a 64-dimensional embedding whose class manifolds are
nothing like isotropic balls. K-Means will do fine on blobs (that is exactly its model) and badly the
moment the structure stops being round. It is also told `k` and forces every point into a cluster — no
notion of noise. I want the first real rung to attack the convexity wall head-on, so I start from the
one fact my eye actually uses when it picks groups out of a scatter: the groups are *dense* and the
gaps and outliers are *sparse*. Density is the signal. Let me build the simplest honest formalization
of exactly that, and nothing fancier.

The crudest measure of "how dense is it around a point `p`" is a count: how many points sit within some
radius. Fix a radius `eps` and define the `eps`-neighborhood `N_eps(p) = { q : dist(p,q) ≤ eps }`; the
local density at `p` is `|N_eps(p)|`. This is metric-agnostic — with Euclidean `dist` the neighborhood
is a ball — and it is the only primitive I will need. Here the harness has already StandardScaled the
data, so `eps` is measured in standard-deviation units and one radius means roughly the same thing
across the feature axes; that scaling is what makes a *fixed* `eps` even thinkable.

First instinct: a cluster is a region where every point has high density — every cluster point `p` has
`|N_eps(p)| ≥ MinPts`. Test it on a picture and it breaks immediately. A point deep in a blob's
interior has many neighbors within `eps`; a point on the blob's *rim* has a half-empty neighborhood,
because half its `eps`-ball pokes out into the void. Border points of a genuine cluster have
significantly fewer neighbors than interior points. So if I set `MinPts` high enough to mean "this is
dense," I gnaw every cluster down to its core and throw away the rims; if I lower `MinPts` until rims
qualify, the bar is now so low it also admits noise. No single count threshold includes borders and
excludes noise, because border density is an artifact of being at the edge, not a property of the
cluster. That is the wall the whole construction has to climb.

The escape is to stop demanding high density of *every* cluster point and demand it of every cluster
point's *vicinity*. A border point may be sparse itself, provided it lies within `eps` of a genuinely
dense point that vouches for it. Split the points: call `p` **core** if it clears the bar itself,
`|N_eps(p)| ≥ MinPts` — the dense interior — and call a point that fails the bar but lies within `eps`
of a core point a **border** point, riding on a core neighbor's density. Make this one-directional and
precise: `p` is *directly density-reachable* from `q` if `p ∈ N_eps(q)` and `q` is core. The relation
is symmetric between two core points but asymmetric at the core/border seam — a core `q` reaches its
border neighbor `p`, but the border `p`, failing the core test, cannot vouch for anyone. That asymmetry
is the formal statement that density flows outward from the dense interior to the sparse rim.

One step of "directly density-reachable" reaches one ring out. To trace a whole arbitrary-shaped region
I chain it — exactly the "walk in small steps" connectivity idea, made concrete as a walk through core
points: `p` is *density-reachable* from `q` if there is a chain `q = p_1, p_2, …, p_n = p` with each
`p_{i+1}` directly density-reachable from `p_i`. Every link except possibly the last endpoint must be
core. This is transitive by construction, and shape-free: at no point did I ask the region to be a ball
or a Voronoi cell — I only ever asked the next point to be a dense neighbor of the current one. So I can
thread along a curved half-moon or around one cluster wrapped in another. Arbitrary shape falls out of
"follow the dense backbone" — which is precisely the property K-Means cannot have. Two border points on
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
labeled `-1`). The rung's real content is therefore the two parameters DBSCAN exposes, `eps` and
`min_samples`, because the abstract method is only as good as the radius I pick, and a global `eps` is
the method's one genuine weakness.

Take `min_samples` first — the density-smoothing count, the bar for "dense enough to be core." Its job
is to stabilize the density estimate: too small and a couple of stray points clumping look like a
cluster; large enough and the estimate is steady. The well-known empirical fact is that the result is
*insensitive* to it over a reasonable range, so I can essentially fix it. For low-dimensional
StandardScaled data the standard choice is a small constant on the order of ten — the sklearn DBSCAN
demonstration uses `min_samples = 10` on StandardScaled 2-D blobs — so for the low-D settings (blobs,
moons; ≤3 features) I pin `min_samples = 10`.

`eps` is the parameter that actually decides the clustering, and it is where the global-threshold
weakness lives. For the low-D settings I can read it off the data's scale, but I have a concrete
calibration anchor: on StandardScaled 2-D the demo `eps = 0.3` works for tight blobs
(`cluster_std ≈ 0.4`), but *this* task's blobs run much looser (`cluster_std` up to 1.5) and at
`eps = 0.3` neighboring looser blobs merge; tightening to roughly `eps = 0.22` keeps them apart while
still connecting each blob's interior. So for `n_features ≤ 3` I set `eps = 0.22`, `min_samples = 10` —
a single fixed pair that the StandardScaler makes meaningful across blobs and moons alike. I am aware
this is a compromise: one global `eps` cannot be simultaneously tight for a dense blob and loose for a
diffuse one, so the varying `cluster_std` on blobs is exactly the case a single radius handles least
gracefully — that is the structural cost of a global threshold, and I accept it for the first rung
because the alternative (re-deriving per-cluster radii) is the next method's job, not this one's.

The high-dimensional setting (digits, 64 features) is where I expect the most trouble and where I refuse
to hard-code a number, so I derive `eps` from the data via the classical `k`-distance knee. For each
point compute the distance to its `k`-th nearest neighbor and sort those values descending: points deep
in a dense region have a small `k`-distance, points out in the sparse void have a large one, so the
sorted curve starts high (noise) and falls to low (cluster), with a *knee* at the transition. The
`k`-distance at that knee is the right `eps` — it is the density of the thinnest region I am still
willing to call a cluster. I detect the knee Kneedle-style: treat the sorted curve as points, draw the
chord from its first to its last point, and take the curve point of maximum perpendicular distance from
that chord; its `k`-distance is `eps`. For the high-D bar I scale `min_samples` with dimension
(roughly `2·dim`, capped to a small constant so it stays sane on a few-thousand-point set) and use the
matching `k`. I keep one convention straight: the `k`-th nearest neighbor excludes the point itself,
while the core test counts the self-point, so the `k` in the distance curve is `min_samples − 1` in
spirit; I let `NearestNeighbors` return `k+1` and read the last column.

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
methods that blobs reward, and silhouette will be dragged by the noise label. **Digits** is the one I am
genuinely worried about: in 64 dimensions distances concentrate (every pairwise distance creeps toward
the same value), so the `k`-distance curve flattens and its knee is ill-defined; the most likely
failure is that the derived `eps` is too small, almost everything is declared noise, and the labeling
collapses toward a single non-noise cluster — which the harness scores `silhouette = −1.0` and near-zero
ARI/NMI. If that happens, it is not a bug in DBSCAN; it is the diagnosis that *global density
thresholding does not survive the curse of dimensionality*, and it points straight at the next rung. So
my falsifiable expectation entering the run is: moons strong, blobs solid-but-capped, digits at serious
risk of a degenerate collapse — and whichever way digits lands is the number the next method has to
answer for.
