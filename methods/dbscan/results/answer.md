# DBSCAN, distilled

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) defines a cluster as a
maximal set of *density-connected* points and finds clusters as the connected components of
the dense regions of the data, separated by sparse regions. It needs no number of clusters in
advance, recovers clusters of arbitrary shape, labels outliers as noise, and — issuing exactly
one neighborhood query per point — runs efficiently on a spatial index.

## Problem it solves

Cluster a large `k`-dimensional spatial database with: (1) minimal domain knowledge (no `k`
given), (2) arbitrary cluster shapes (not just convex blobs), (3) efficiency on large data (no
all-pairs `O(n^2)` cost), and (4) explicit handling of noise/outliers. Partitioning methods
(k-means, k-medoid, CLARANS) need `k`, produce convex Voronoi cells, and force every point into
a cluster; hierarchical/connectivity methods get shape but cost `O(n^2)` or need an unsettable
termination threshold; grid-histogram methods get shape but are heavy and cell-size sensitive.

## Key idea

Estimate local density by counting points within a radius and threshold it. Fix a radius `Eps`
and a count `MinPts`. The **`Eps`-neighborhood** of a point is

  `N_Eps(p) = { q ∈ D | dist(p, q) ≤ Eps }`   (any metric; the metric shapes the neighborhood).

- A point is a **core point** if `|N_Eps(p)| ≥ MinPts` (dense enough; the self-point is counted).
- `p` is **directly density-reachable** from `q` if `p ∈ N_Eps(q)` and `q` is core. (Symmetric
  for two core points; asymmetric when `q` is core and `p` is a non-core *border* point.)
- `p` is **density-reachable** from `q` if there is a chain `q = p_1, …, p_n = p`, `n ≥ 2`,
  with each `p_{i+1}` directly density-reachable from `p_i`. Transitive; not symmetric (every
  point with an outgoing step must be core; only the final endpoint may be border). This chaining
  through core points is what gives arbitrary cluster shape — no convexity is assumed.
- `p` and `q` are **density-connected** if some point `o` density-reaches both. Symmetric; in any
  nontrivial same-cluster case that witness is core, which lets two border points on opposite
  rims count as one cluster.

A **cluster** `C` (wrt `Eps, MinPts`) is a non-empty subset of `D` that is

  (1) **maximal**: if `p ∈ C` and `q` is density-reachable from `p`, then `q ∈ C`; and
  (2) **connected**: every `p, q ∈ C` are density-connected.

**Noise** = points in no cluster. Every cluster has `≥ MinPts` points (its seed core point
already has that many neighbors, all in the cluster).

This is the minimum-density-level-set view: `|N_Eps(p)|` is a uniform-kernel density estimate
with bandwidth `h = Eps`; core points are where the estimate exceeds the level `MinPts/n`;
cluster backbones are connected components of that super-level set, with adjacent non-core
sample points absorbed as border points.

## Two lemmas (the algorithm in disguise)

**Lemma 1.** If `p` is a core point, then `O = { o | o density-reachable from p }` is a cluster.
*Proof.* `p ∈ O` (p is directly density-reachable from itself ⇒ non-empty); maximality from
transitivity of density-reachability; connectivity since every member is density-reachable from
the single core point `p`. ∎

**Lemma 2.** If `C` is a cluster and `p ∈ C` is core, then `C = { o | o density-reachable from
p }`. *Proof.* `⊇`: any such `o` is in `C` by maximality. `⊆`: for `q ∈ C`, connectivity gives a
point `o` reaching both `p` and `q`. The chain `o → … → p` has core at both ends: `p` is core,
and `o` is either `p` or must be core to start the first direct step. All interior sources are
core, so each direct step can be reversed and `o` is density-reachable from `p`; chaining
`p → … → o → … → q` gives `q` density-reachable from `p`. ∎

So a cluster is exactly the density-reachable set of *any* of its core points ⇒ to find a
cluster, seed at a core point and flood-fill its density-reachable set.

## Algorithm

```
for each unprocessed point p in D:
    ExpandCluster(p, next_label) ; if it returns True, advance next_label

ExpandCluster(p, ClId):
    seeds <- regionQuery(p, Eps)              # = N_Eps(p)
    if |seeds| < MinPts:                       # p is not core
        label(p) <- NOISE                      # provisional: may be reclaimed as a border point
        return False
    label every point of seeds <- ClId
    remove p from seeds
    while seeds not empty:
        cur <- seeds.first()
        result <- regionQuery(cur, Eps)
        if |result| >= MinPts:                 # cur is core -> propagate
            for q in result:
                if label(q) in {UNCLASSIFIED, NOISE}:
                    if label(q) == UNCLASSIFIED: seeds.append(q)   # only unprocessed expand
                    label(q) <- ClId           # NOISE -> border point of this cluster
        seeds.remove(cur)
    return True
```

Border points are absorbed but never expanded (no density to propagate). `regionQuery` is
called **exactly once per point**, so the result is order-independent except that a border
point reachable from two clusters joins whichever reaches it first (rare, immaterial).

## Complexity

Runtime `= O(n · Q)`, `Q` the cost of one region query. Without an index `Q = O(n)` ⇒ `O(n^2)`.
With an R\*-tree and *small* region queries (`Eps`-neighborhoods small vs. the data space) a
query is roughly `O(log n)` on average ⇒ roughly `O(n log n)` average runtime. The `O(log n)`
per-query figure is an informal average-case observation for small queries on the index, not a
worst-case guarantee. Space `O(n)` for the labels (plus the index).

## Choosing the parameters

- **`MinPts`** smooths the density estimate; the clustering is largely insensitive to it, so fix
  it: `MinPts = 4` for 2-D data, `MinPts ≈ 2·dim` in general, raised for very noisy data or data
  with many duplicates. This leaves essentially one free parameter.
- **`Eps`** from the **sorted `k`-dist graph**: for each point take the distance to its `k`-th
  nearest neighbor, sort all points in descending order, and read `Eps` off the first
  "valley"/knee — points left of the knee (large `k`-dist, sparse) become noise, points right of
  it (dense) cluster. Choose `Eps` as small as possible. Off-by-one to respect: the `k`-th
  *nearest neighbor* excludes the query point but `N_Eps` includes it, so the graph's `k`
  corresponds to `MinPts = k + 1`.
- **Red flags** for a bad `Eps`: noise fraction outside ~1–30% (near-zero ⇒ `Eps` too large),
  or the largest connected component exceeding ~20–50% of clustered points (⇒ `Eps` too large,
  dense substructure merged — reduce `Eps`).
- Global `Eps, MinPts` (one pair for all clusters) is the simple default; its only cost is that
  two clusters of different density lying within `Eps` of each other can merge, fixable by
  re-running the merged region with a higher `MinPts`.

## Working code

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors

NOISE = -1


class DBSCAN(BaseEstimator, ClusterMixin):
    """Density-based clustering. Core points (>= min_samples neighbors within eps)
    form the dense backbone; clusters are connected components of core points with
    their non-core eps-neighbors absorbed as border points; the rest is noise (-1)."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps                      # Eps: neighborhood radius / KDE bandwidth
        self.min_samples = min_samples      # MinPts: core-point density threshold

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # One region query per point, via the spatial index. radius_neighbors
        # returns N_Eps(p) including p itself (dist(p,p)=0 <= eps) -- the count
        # the core test uses.
        nn = NearestNeighbors(radius=self.eps, metric="euclidean").fit(X)
        neighborhoods = nn.radius_neighbors(X, return_distance=False)
        n_neighbors = np.array([len(nb) for nb in neighborhoods])
        is_core = n_neighbors >= self.min_samples          # |N_Eps(p)| >= MinPts

        labels = np.full(n, NOISE, dtype=np.intp)
        cluster_id = 0
        for i in range(n):
            # non-core points cannot seed a cluster (Lemma 2); skip assigned points
            if labels[i] != NOISE or not is_core[i]:
                continue
            labels[i] = cluster_id
            stack = [i]                                     # ExpandCluster (stack-based)
            while stack:
                p = stack.pop()
                if not is_core[p]:                         # only core points propagate
                    continue
                for q in neighborhoods[p]:                  # q directly density-reachable from p
                    if labels[q] == NOISE:
                        labels[q] = cluster_id
                        if is_core[q]:
                            stack.append(q)                 # only core points expand
            cluster_id += 1

        # Points never reached from a core point stay at the noise label -1.
        self.labels_ = labels
        self.n_clusters_ = cluster_id
        self.core_sample_indices_ = np.where(is_core)[0]
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


def custom_distance(x, y):
    # Euclidean distance: its Eps-ball is the shape of the Eps-neighborhood N_Eps.
    return np.sqrt(np.sum((x - y) ** 2))
```

This matches the canonical scikit-learn implementation: bulk radius-neighbor queries, a boolean
core mask `n_neighbors >= min_samples`, and a flood-fill (`dbscan_inner`) that labels connected
components of core points while absorbing — but not expanding from — border points, leaving
unreached points at the noise label `-1`.

## Relation to prior methods

- **k-means / k-medoid (PAM) / CLARANS**: partitioning by nearest representative ⇒ convex
  Voronoi-cell clusters, a required `k`, and no noise. DBSCAN drops all three by replacing
  "nearest representative" with "connected dense region."
- **Hierarchical / Ejcluster (García et al. 1994)**: the "walk in small steps" connectivity
  recovers shape but at `O(n^2)`; DBSCAN realizes the same connectivity through density-reachable
  chains but with one indexed region query per point.
- **Grid-histogram (Jain 1988)**: recovers shape via occupancy but is space/time heavy and
  cell-size sensitive; DBSCAN reads density pointwise via `N_Eps` and needs no grid.
