# Context: density-based clustering circa 2013

## Research question

We are given an unlabeled set of points `X = {x_1, ..., x_n}` in a metric space and want to
partition them into groups that reflect the underlying structure of the data — and, crucially,
to leave the points that belong to no group unassigned as *noise*. The structures we care about
are not the convex, isotropic blobs that centroid methods assume: they can be elongated,
non-convex, nested, and — the hard part — they can sit at *very different local densities* in
the same dataset. One region may be a tight, dense knot; another a diffuse cloud; a third a
thin filament threading between them.

The precise difficulty is this. Density-based methods model a cluster as a region where points
are packed more tightly than in the surrounding space, and noise as the sparse remainder. To
make that operational you have to fix a density level: how dense is "dense enough" to count as
cluster rather than noise. A single global density level is a single knob, and one knob cannot
be right everywhere at once. Set it high and the diffuse cloud dissolves into noise; set it low
and the tight knot merges with its neighbors through whatever sparse bridge connects them. What
a solution would have to achieve is a clustering that draws *different* groups at *different*
density levels — picking up the dense knot at one level and the diffuse cloud at another, in a
single output — while needing as little hand-tuning as possible, ideally one intuitive
parameter rather than a brittle radius the analyst must guess. It should also produce a flat
partition (one label per point, plus noise) that downstream tasks can consume, not only an
exploratory diagram a human must read.

## Background

The formal model underneath all of this is Hartigan's notion of *density-contour clusters*
(Hartigan 1975). Suppose the data are drawn from an unknown probability density `f`. For a
density level `lambda`, the level set is `{x : f(x) >= lambda}`; its maximal connected
components are the density-contour clusters at level `lambda`. As `lambda` increases, the level
set shrinks: a component stays connected but loses its low-density fringe, or it splits into two
denser components, or it vanishes entirely. The nested family of these components over all
`lambda` is the *density-contour tree*. This model is attractive for three reasons: it gives
noise a principled meaning (points lying below the threshold, in non-dense regions, simply do
not belong to any cluster — observations are not forced into groups); it lets clusters take
arbitrary shapes (a connected component need not be convex); and it represents clusters of
different densities as nodes at different levels of the same tree. The catch is that `f` is
unknown, so every practical method estimates it, and estimates the connected components of its
level sets, in some way — and most density-based algorithms commit to a *single* level `lambda`,
collapsing the whole tree to one horizontal slice.

A few diagnostic facts about the design space are worth stating up front, because they are what
make the single-level approach brittle. First, in single-linkage hierarchical clustering, a
single thin bridge of points — even spurious noise — suffices to merge two otherwise separate
dense clusters; this is the classic *chaining* pathology. Second, in any density hierarchy a
horizontal cut corresponds to one global density threshold, so it provably cannot separate two
clusters whose internal densities straddle the cut: the denser one survives the cut intact while
the sparser one has already dissolved or has not yet separated. Third — observed repeatedly in
the literature on density-based methods — the result is highly sensitive to the radius
parameter, and there is no global radius that simultaneously resolves clusters of widely varying
density. These are not failures of a particular implementation; they are consequences of using
one threshold for a structure that lives at many.

The relevant estimation machinery is K-nearest-neighbor density. The distance from a point to
its `K`-th nearest neighbor, under the convention used in the density model here where the point
itself is counted as the first neighbor, is small in dense regions and large in sparse ones. Its
reciprocal is a (non-normalized) estimate of the local density, with `K` acting as a smoothing
factor. A larger `K` smooths more; the estimate is used here only to discriminate dense from
non-dense, so it need not be an accurate density — which makes `K` an unusually forgiving
parameter.

Finally, *excess of mass* (Hartigan 1987; Muller & Sawitzki 1991, revisited by Stuetzle &
Nugent 2010) is a way to score how prominent a density-contour cluster is. For a cluster `C`
that first appears at level `lambda_min(C)`, its excess of mass is `E(C) = integral_{x in C} (
f(x) - lambda_min(C) ) dx` — the volume of density sitting *above* the level at which `C` was
born. Intuitively, a cluster that stays present over a long range of `lambda` after it appears,
holding many points, accumulates a large excess of mass and is "prominent"; one that
immediately dissolves accumulates little. The catch is nesting: the excess of a parent cluster
contains the excesses of its descendants, so raw excess of mass increases along a branch and
cannot by itself decide whether the parent or its children are the better flat clusters.

## Baselines

**Single-linkage / minimum-spanning-tree hierarchical clustering (Johnson 1967; Jain & Dubes
1988).** Build a complete graph on the points with edge weights equal to pairwise distances,
then remove edges in *decreasing* order of weight and track the connected components; the nested
components form a dendrogram. Equivalently and far more cheaply, compute a minimum spanning tree
(MST) of the graph and remove its `n-1` edges in decreasing weight order — the MST carries all
the merge information of single-linkage. This is the canonical way to recover a hierarchy of
nested groupings at all distance scales at once. **Limitation:** it has no notion of noise —
every point is in some cluster at every level — and it suffers chaining: a single sparse path of
points merges two dense clusters, because single-linkage joins components on their *closest*
pair of members, so one short bridging edge is enough.

**DBSCAN / DBSCAN\* (Ester, Kriegel, Sander, Xu 1996).** Fix a radius `eps` and a count
`MinPts`. A point `p` is a *core* point if its `eps`-neighborhood `N_eps(p) = {x : d(x,p) <=
eps}` contains at least `MinPts` points. Clusters are the connected components of core points
under the relation "within `eps` of each other"; points that are not reachable are noise. (The
original DBSCAN additionally absorbs *border* points — non-core points inside a core point's
`eps`-ball — into clusters; the cleaner variant DBSCAN\* drops them so that clusters are exactly
connected components of core objects, which keeps a precise level-set interpretation and makes
reachability symmetric.) DBSCAN finds arbitrarily shaped clusters and labels noise, and it runs
in roughly `O(n log n)` with a spatial index. **Limitation:** the radius `eps` is a single
global density threshold. One `eps` cannot capture clusters of differing density at once, and
the result is acutely sensitive to it — Ester et al.'s own recipe for choosing `eps` (read it
off a "valley" in the sorted `k`-distance plot) is interactive and approximate. `MinPts` doubles
as a de-facto minimum cluster size.

**OPTICS (Ankerst, Breunig, Kriegel, Sander 1999).** Rather than commit to one `eps`, OPTICS
produces an ordering of the points and, for each, a *reachability distance*, then visualizes the
result as a bar plot (the *reachability plot*) in which clusters appear as valleys. Two
quantities drive it. The core distance of `p` is the distance to its `MinPts`-th nearest
neighbor (the smallest radius at which `p` is core). The reachability distance of `o` *from* `p`
is `reach(o, p) = max( core-dist(p), d(p, o) )`. **Limitation:** the reachability distance is
*asymmetric* — it is measured from `p`, so `reach(o, p)` and `reach(p, o)` differ — which means
the plot only *approximately* corresponds to DBSCAN's clusters, and the relationship between the
two is not exact. Extracting a flat clustering from the plot is itself a separate heuristic
(threshold the bars, or detect steep up/down areas); thresholding the bars at one level is once
again a single global cut, with the varying-density limitation back in force.

## Evaluation settings

The natural yardsticks are unlabeled datasets with known ground-truth structure spanning the
geometries that stress different assumptions:

- **Isotropic Gaussian blobs** — several convex, roughly equal-density clusters; the easy case
  every method should handle, and a sanity check that a density method does not over-fragment a
  convex cluster.
- **Interleaving half-moons** — two non-convex, interlocking shapes; centroid methods fail here
  by construction, so this isolates whether a method respects connectivity rather than convexity.
- **High-dimensional real data** — e.g. handwritten-digit feature vectors (tens of features,
  ten classes); tests behavior when the geometry is high-dimensional and clusters have unequal
  density and size.

Quality is measured against the known labels with the Adjusted Rand Index (agreement of the
predicted partition with the truth, corrected for chance) and Normalized Mutual Information
(shared information between the two partitions), and intrinsically with the silhouette
coefficient (within-cluster tightness versus between-cluster separation). The protocol fixes the
preprocessing (standardization) and feeds the raw feature matrix `X` to the clustering estimator;
the number of clusters is not given to a method that claims to discover it.

## Code framework

The method will be packaged as a clustering *estimator* in the standard `fit(X) -> labels_` /
`predict` style used by the rest of the pipeline. `fit(X)` consumes an `(n_samples, n_features)`
array and must populate an integer label per sample (with a reserved label for noise);
`predict` returns those labels. What already exists is the estimator base classes, a standard
scaler, a distance helper, and — since everything here is built on geometric proximity — the
pairwise-distance and nearest-neighbor primitives. What rule turns the points and their
distances into labels is exactly what is to be designed, so the assignment is left as one empty
slot.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import pdist, squareform


class CustomClustering(BaseEstimator, ClusterMixin):
    """A clustering estimator. fit(X) must set self.labels_ (one integer label
    per sample; a reserved value marks noise); predict(X) returns the labels.
    The rule that turns the points into labels is what we are about to design."""

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters          # may be unknown a priori
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        # X: (n_samples, n_features).
        # Geometric primitives are available:
        #   D = squareform(pdist(X))          # pairwise distances
        # TODO: the assignment rule we will design -- turn X into self.labels_.
        pass

    def predict(self, X):
        if self.labels_ is None:
            self.fit(X)
        return self.labels_


def custom_distance(x, y):
    # The base metric defining "closeness" of two points; the geometry the
    # assignment rule sees is built on top of it.
    return np.sqrt(np.sum((x - y) ** 2))
```
