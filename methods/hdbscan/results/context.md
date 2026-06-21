# Context: density-based clustering circa 2013

## Research question

We are given an unlabeled set of points `X = {x_1, ..., x_n}` in a metric space and want to
partition them into groups that reflect the underlying structure of the data, leaving the points
that belong to no group unassigned as *noise*. The structures of interest are not the convex,
isotropic blobs that centroid methods assume: they can be elongated, non-convex, nested, and
sit at very different local densities in the same dataset. One region may be a tight, dense
knot; another a diffuse cloud; a third a thin filament threading between them.

Density-based methods model a cluster as a region where points are packed more tightly than in
the surrounding space, and noise as the sparse remainder. To make that operational one fixes a
density level: how dense is "dense enough" to count as cluster rather than noise. The question
here is how to produce, from a single run with as little hand-tuning as possible, a flat
partition (one label per point, plus noise) that recovers clusters living at differing local
densities in the same dataset.

## Background

The formal model underneath all of this is Hartigan's notion of *density-contour clusters*
(Hartigan 1975). Suppose the data are drawn from an unknown probability density `f`. For a
density level `lambda`, the level set is `{x : f(x) >= lambda}`; its maximal connected
components are the density-contour clusters at level `lambda`. As `lambda` increases, the level
set shrinks: a component stays connected but loses its low-density fringe, or it splits into two
denser components, or it vanishes entirely. The nested family of these components over all
`lambda` is the *density-contour tree*. This model gives noise a principled meaning (points
lying below the threshold, in non-dense regions, simply do not belong to any cluster —
observations are not forced into groups); it lets clusters take arbitrary shapes (a connected
component need not be convex); and it represents clusters of different densities as nodes at
different levels of the same tree. In practice `f` is unknown, so every method estimates it, and
estimates the connected components of its level sets, in some way; most density-based algorithms
commit to a single level `lambda`, one horizontal slice of the tree.

Two structural facts about this design space are worth stating. In single-linkage hierarchical
clustering, components are joined on their closest pair of members, so a thin bridge of points
suffices to merge two otherwise separate dense clusters (the classic *chaining* behavior). And
in any density hierarchy a horizontal cut corresponds to one global density threshold: of two
clusters whose internal densities straddle the cut, the denser one survives the cut intact while
the sparser one may already have dissolved or not yet separated.

The relevant estimation machinery is K-nearest-neighbor density. The distance from a point to
its `K`-th nearest neighbor, under the convention used in the density model here where the point
itself is counted as the first neighbor, is small in dense regions and large in sparse ones. Its
reciprocal is a (non-normalized) estimate of the local density, with `K` acting as a smoothing
factor: a larger `K` smooths more. The estimate is used here only to discriminate dense from
non-dense, so it need not be an accurate density.

Finally, *excess of mass* (Hartigan 1987; Muller & Sawitzki 1991, revisited by Stuetzle &
Nugent 2010) is a way to score how prominent a density-contour cluster is. For a cluster `C`
that first appears at level `lambda_min(C)`, its excess of mass is `E(C) = integral_{x in C} (
f(x) - lambda_min(C) ) dx` — the volume of density sitting *above* the level at which `C` was
born. A cluster that stays present over a long range of `lambda` after it appears, holding many
points, accumulates a large excess of mass and is "prominent"; one that immediately dissolves
accumulates little. Because a parent cluster contains its descendants, the excess of a parent
contains the excesses of its children, so raw excess of mass increases along a branch.

## Baselines

**Single-linkage / minimum-spanning-tree hierarchical clustering (Johnson 1967; Jain & Dubes
1988).** Build a complete graph on the points with edge weights equal to pairwise distances,
then remove edges in *decreasing* order of weight and track the connected components; the nested
components form a dendrogram. Equivalently and far more cheaply, compute a minimum spanning tree
(MST) of the graph and remove its `n-1` edges in decreasing weight order — the MST carries all
the merge information of single-linkage. This recovers a hierarchy of nested groupings at all
distance scales at once. Every point is in some cluster at every level: single-linkage joins
components on their *closest* pair of members, so one short bridging edge merges two groups.

**DBSCAN / DBSCAN\* (Ester, Kriegel, Sander, Xu 1996).** Fix a radius `eps` and a count
`MinPts`. A point `p` is a *core* point if its `eps`-neighborhood `N_eps(p) = {x : d(x,p) <=
eps}` contains at least `MinPts` points. Clusters are the connected components of core points
under the relation "within `eps` of each other"; points that are not reachable are noise. (The
original DBSCAN additionally absorbs *border* points — non-core points inside a core point's
`eps`-ball — into clusters; the cleaner variant DBSCAN\* drops them so that clusters are exactly
connected components of core objects, which keeps a precise level-set interpretation and makes
reachability symmetric.) DBSCAN finds arbitrarily shaped clusters and labels noise, and it runs
in roughly `O(n log n)` with a spatial index. The radius `eps` is a single global density
threshold; Ester et al.'s recipe for choosing it reads a "valley" off the sorted `k`-distance
plot. `MinPts` doubles as a de-facto minimum cluster size.

**OPTICS (Ankerst, Breunig, Kriegel, Sander 1999).** Rather than commit to one `eps`, OPTICS
produces an ordering of the points and, for each, a *reachability distance*, then visualizes the
result as a bar plot (the *reachability plot*) in which clusters appear as valleys. Two
quantities drive it. The core distance of `p` is the distance to its `MinPts`-th nearest
neighbor (the smallest radius at which `p` is core). The reachability distance of `o` *from* `p`
is `reach(o, p) = max( core-dist(p), d(p, o) )`. This quantity is measured from `p`, so
`reach(o, p)` and `reach(p, o)` differ in general. A flat clustering is read off the plot by a
separate procedure: threshold the bars at one level, or detect steep up/down areas.

## Evaluation settings

The natural yardsticks are unlabeled datasets with known ground-truth structure spanning the
geometries that stress different assumptions:

- **Isotropic Gaussian blobs** — several convex, roughly equal-density clusters; the easy case
  every method should handle, and a sanity check that a density method does not over-fragment a
  convex cluster.
- **Interleaving half-moons** — two non-convex, interlocking shapes; isolates whether a method
  respects connectivity rather than convexity.
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
