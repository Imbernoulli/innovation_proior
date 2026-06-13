# Context: clustering large spatial databases (circa mid-1990s)

## Research question

Spatial database systems — earth-observation archives, astronomical catalogs, protein and
crystallography data, satellite imagery — accumulate enormous numbers of objects, each
carrying a position in some `k`-dimensional feature space, and a recurring task is *class
identification*: grouping the objects into meaningful subclasses purely from their spatial
distribution, with no labels. Automating this matters because the databases are far too large
to inspect by hand and the structure is exactly what downstream knowledge discovery wants
(e.g. picking out the houses strung along a river in an earth-observation database).

A clustering algorithm fit for this setting has to meet three demands *at once*, and that
simultaneity is the whole difficulty:

1. **Minimal domain knowledge to set its parameters.** On a database we are exploring for the
   first time, we typically do *not* know how many groups there are, and any parameter the
   user is asked to supply must be one a non-expert can actually choose.
2. **Discovery of clusters of arbitrary shape.** Spatial clusters are not tidy balls — they
   can be spherical, linear, elongated, drawn-out, sinuous, or wrapped around one another,
   because the physical processes that lay points down in space produce shapes like these. An
   algorithm that can only return round groups will mis-segment real spatial data.
3. **Good efficiency on large databases**, meaning databases of far more than a few thousand
   objects — so an all-pairs `O(n^2)` cost is already too expensive.

On top of these three: real spatial data contains *noise* — outliers belonging to no group —
and a usable method must be able to leave such points out rather than forcing every point into
some cluster. The prevailing methods of the day each satisfy a subset of these and miss the
rest; closing the gap between "satisfies some" and "satisfies all four" is the problem.

## Background

The empirical fact every clustering method is implicitly chasing is simple to state and easy
to see by eye. Look at a scatter of spatial points containing a few groups plus scattered
outliers, and what makes the groups perceptible is that **within a group the points sit close
together — the local density of points is markedly higher than in the surrounding space — and
the regions between groups, and the regions where the outliers live, are comparatively
sparse.** The groups are the dense patches; the gaps between them, and the thin scatter of
outliers, are the low-density background. This is visible in any such scatter before any
algorithm is written, and it is the raw phenomenon a spatial clustering method has to capture.

A handful of standard ingredients are on the table for building such a method:

- **A distance function `dist(p, q)`** on the `k`-dimensional feature space. Everything about
  "closeness" is mediated by this metric, and crucially the *shape* of a fixed-radius
  neighborhood follows the metric: with Euclidean distance a neighborhood of radius `r` is a
  ball, with the Manhattan distance it is a diamond/rectangle. Nothing forces Euclidean — an
  appropriate `dist` can be chosen per application, and a method ought to inherit whatever
  metric the domain supplies.

- **Spatial access methods / spatial indexes.** Large spatial databases are stored in indexes
  built for exactly the kind of geometric query clustering needs. The **R\*-tree** (Beckmann,
  Kriegel, Schneider & Seeger 1990) is the workhorse: a balanced tree of nested bounding
  rectangles, height `O(log n)`, that answers a *region query* — "return all stored objects
  inside this query region" — by descending only the few subtrees whose rectangles overlap
  the region. When the query region is small relative to the whole data space, only a limited
  number of root-to-leaf paths are touched, so such a query is cheap on average (Brinkhoff,
  Kriegel, Schneider & Seeger 1994 study efficient processing of these spatial queries). A
  method that can express its work as a stream of small region queries can ride this index
  instead of scanning the whole database for every object.

- **The notion of a representative and an objective.** Much of clustering is framed as
  choosing `k` representatives and a within-cluster cost `E = Σ_clusters Σ_{p∈cluster}
  dist(p, rep)` to minimize — the lever the partitioning methods below all pull.

- **Cluster-validity scores.** The **silhouette coefficient** (Kaufman & Rousseeuw 1990) rates
  a clustering by, for each point, comparing its mean distance to its own cluster against its
  mean distance to the nearest other cluster; averaging gives a score in `[-1, 1]` used to
  *select* the number of groups by trying several and keeping the best — but each trial is a
  full clustering, so this selection is only as cheap as the underlying method run many times.

## Baselines

The prior methods a new spatial-clustering algorithm would be measured against, with the core
idea, the actual objective/procedure, and where each one stalls.

**Partitioning around representatives — k-means and k-medoid (Lloyd 1957; MacQueen 1967;
Kaufman & Rousseeuw 1990).** Fix the number of groups `k` up front. Choose `k` representatives
and assign every object to its nearest one, minimizing `E = Σ_{j=1}^{k} Σ_{p∈C_j} dist(p,
rep_j)`. In *k-means* the representative of a cluster is its gravity center (the coordinate
mean); in *k-medoid* it is an actual data object, the *medoid*, the most centrally located
member. The procedure is two interleaved steps iterated to convergence: given representatives,
assign each object to the closest one; given the assignment, recompute each representative to
minimize the cost. Because every object goes to its single nearest representative, the
resulting partition is exactly the **Voronoi diagram** of the representatives, and each cluster
is contained in one Voronoi cell. **Gaps:** (a) `k` must be supplied in advance, which on an
unexplored database is precisely the unknown; (b) a Voronoi cell is convex, so every returned
cluster is convex — an elongated, sinuous, or nested group cannot be represented and will be
sliced across several cells; (c) there is no notion of an outlier — every object is forced into
some cluster, and a far-off point drags its representative toward itself, distorting the group.

**k-medoid for larger databases — CLARANS (Ng & Han 1994).** Clustering Large Applications
based on RANdomized Search improves the older k-medoid solvers (PAM, CLARA of Kaufman &
Rousseeuw 1990). It recasts the search as moving through a graph whose nodes are sets of `k`
medoids and whose edges connect node-sets differing in a single medoid; it does a randomized
hill-climb — from the current set, sample a bounded number of single-medoid swaps, move to one
that lowers the objective `E`, and restart a few times to escape poor local optima. This is more
effective and more efficient than evaluating every swap as PAM does. To address "what is the
*natural* number of groups," Ng & Han run CLARANS once for each `k` from 2 to `n`, score each
clustering by its silhouette coefficient, and keep the `k` with the best score. **Gaps:** (a)
it is still a partitioning method, so it inherits the convex-Voronoi-cell limitation — clusters
of arbitrary shape are out of reach; (b) the run time is prohibitive on large databases: a
single CLARANS run is roughly quadratic in `n`, and the silhouette-driven search for `k`
multiplies that by a sweep over candidate `k` values; (c) it assumes all objects reside in
main memory simultaneously, which fails for large databases; (d) it has no explicit noise
model — every object is assigned to its closest medoid.

**Hierarchical methods (agglomerative / divisive; Kaufman & Rousseeuw 1990).** Build a
*dendrogram*: agglomerative methods start with every object its own group and repeatedly merge
the two closest groups; divisive methods start with one group and repeatedly split. No `k` is
required as input — but a *termination condition* is, telling the process when to stop merging
or splitting, e.g. a threshold `D_min` on the inter-cluster distance below which two groups are
considered one. **Gap:** that condition is hard to set — it must be small enough to keep
genuinely separate groups apart, yet large enough not to chop a single group into pieces, and
no single value does both across a database with groups of differing tightness. A recent
variant, *Ejcluster* (García, Fdez-Valdivia, Cortijo & Molina 1994), sidesteps the termination
problem with a connectivity idea — two objects belong to the same group if one can "walk" from
one to the other by a sequence of sufficiently small steps — and is reported very effective on
non-convex groups, deriving its own stopping point; **but** its distance computation is
`O(n^2)` because it inspects pairs of points, acceptable only for small datasets such as
character recognition, not for large spatial databases.

**Grid / density-histogram methods (Jain 1988).** Partition the feature space into a grid of
non-overlapping cells and build a multidimensional histogram of how many objects fall in each
cell; cells with relatively high counts are candidate cluster centers, and the boundaries
between groups fall in the "valleys" of the histogram. This *can* recover groups of arbitrary
shape, since it reads structure off the occupancy pattern rather than fitting representatives.
**Gaps:** the space and run time to store and search a multidimensional histogram grow
enormously with dimension, and the result depends crucially on the chosen cell size — too
coarse and groups merge, too fine and a group fragments — which is itself a parameter the user
must guess.

Taken together: the partitioning methods need `k` and return convex groups with no notion of
noise; the hierarchical and connectivity methods avoid `k` but cost `O(n^2)` or need a
termination threshold no single value satisfies; the grid-histogram methods recover shapes but
are heavy in space and time and acutely sensitive to cell size. Each clears some of the four
demands and trips on the others; none clears all four together.

## Evaluation settings

The natural yardsticks for a new spatial-clustering method, all available before it exists:

- **Synthetic 2-D sample databases** chosen to stress shape and noise: one with several
  ball-shaped groups of *significantly differing sizes*; one with several *non-convex* groups
  (curved, interleaving shapes); one with groups of differing shape and size *plus added
  noise*. Because different families of clustering algorithm share no common quantitative
  accuracy measure, the standard comparison on these is *visual inspection* — plotting each
  discovered group in a distinct color and judging whether the perceived structure was
  recovered and whether noise was correctly set aside.
- **Real spatial data — the SEQUOIA 2000 benchmark** (Stonebraker, Frew, Gardels & Meredith
  1993), a database representative of Earth-science tasks; its point dataset holds the
  Californian names of landmarks from the U.S. Geological Survey's Geographic Names
  Information System together with their coordinates (~62,584 points, ~2.1 MB), with subsets
  of increasing size used as the natural scaling test.
- **Protocol:** the result should be independent of the order in which objects are visited;
  efficiency is measured as wall-clock run time versus the number of points (to expose the
  growth rate), and effectiveness by whether the groups a human sees — including the correct
  handling of noise — are the groups returned.

## Code framework

The method will be packaged as a clustering *estimator* in the standard
`fit` → `labels_` / `predict` style: `fit(X)` consumes an `(n_samples, n_features)` array and
must populate an integer label per sample; `predict` returns those labels. The substrate that
already exists is the estimator base classes, a standard scaler, a metric helper, and — because
the data lives in spatial indexes — a nearest-neighbor structure that answers geometric
queries. What rule turns the points into labels is exactly what is to be designed, so the
assignment is left as one empty slot.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors


class CustomClustering(BaseEstimator, ClusterMixin):
    """A clustering estimator. fit(X) must set self.labels_ (one integer label
    per sample); predict(X) returns labels. The rule that assigns the labels is
    what we are about to design."""

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters          # may be unknown a priori
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        # X: (n_samples, n_features).
        # An indexed nearest-neighbor structure is available for geometric queries:
        #   nn = NearestNeighbors(...).fit(X)
        # TODO: the assignment rule we will design -- turn X into self.labels_.
        pass

    def predict(self, X):
        if self.labels_ is None:
            self.fit(X)
        return self.labels_


def custom_distance(x, y):
    # The metric that defines "closeness" between two points; its choice shapes
    # the geometry the assignment rule sees.
    # TODO: pick the distance the assignment rule will use.
    pass
```

The single empty slot is the assignment rule inside `fit` (and the metric it reads closeness
through); the final code fills exactly these stubs.
