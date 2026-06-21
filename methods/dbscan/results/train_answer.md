Spatial databases accumulate unlabeled points in a k-dimensional feature space, and the task is to discover meaningful groups without knowing how many groups exist in advance, while tolerating outliers. Existing partitioning methods such as k-means and k-medoid require the number of clusters as input, force every point into some cluster, and their nearest-representative rule carves space into convex Voronoi cells, so curved, elongated, or interleaved groups are sliced apart. Hierarchical methods avoid specifying k but still need a termination threshold that cannot simultaneously keep loose groups whole and separate tight groups, and the connectivity variants that recover arbitrary shape cost O(n^2). Grid-histogram methods are shape-flexible yet explode in space and time as dimension grows and depend critically on an unguessable cell size. None of these approaches satisfies all four demands at once: no preset cluster count, arbitrary shapes, efficiency on large databases, and explicit handling of noise.

The signal that actually distinguishes groups is the local density contrast visible to the eye: inside a group the points are packed together, while gaps between groups and stray outliers are sparse. DBSCAN, Density-Based Spatial Clustering of Applications with Noise, builds clusters directly from that contrast. It fixes a radius Eps and a count MinPts. The Eps-neighborhood of a point p is the set of database points within distance Eps of p, using whatever distance the domain supplies, so the shape of the neighborhood follows the metric naturally. A point is a core point if its Eps-neighborhood contains at least MinPts points, counting p itself. A point p is directly density-reachable from q if q is core and p lies within Eps of q. Chaining direct steps gives density-reachability: a core point pulls in its neighbors, its core neighbors pull in theirs, and the chain can thread through rivers, arcs, or nested shapes because it never assumes convexity. Two points are density-connected if some core point density-reaches both; this symmetric relation joins even non-core border points on opposite rims of the same cluster. A cluster is a maximal density-connected set, and points belonging to no cluster are labeled noise.

The algorithm is a flood-fill seeded at core points. For each unprocessed point, query its Eps-neighborhood. If the neighborhood is smaller than MinPts, the point is marked noise provisionally, because it may later be claimed as a border point by a nearby core point. If the point is core, start a new cluster label, assign it to the point and its neighbors, and repeatedly pop each newly reached core point, query its neighborhood, and absorb any unclaimed points into the same cluster. Border points are assigned but never expanded, since they do not have enough neighbors to vouch for others. Each point is queried at most once, so with a spatial index such as an R*-tree the overall cost is roughly O(n log n) on average for small neighborhoods, falling back to O(n^2) only when no index is available. MinPts acts as a smoothing constant and can be fixed at about four for two-dimensional data or roughly twice the dimensionality in general; Eps is read from the knee of the sorted k-distance graph, where points to the left of the knee are sparse enough to become noise and points to the right are dense enough to cluster.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors

NOISE = -1


class DBSCAN(BaseEstimator, ClusterMixin):
    """Density-Based Spatial Clustering of Applications with Noise.

    A cluster is a maximal set of density-connected points. Core points
    have at least min_samples neighbors within eps; clusters are the
    connected components of core points, with their non-core eps-neighbors
    absorbed as border points. Everything else is labeled noise (-1).
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # One region query per point, using the spatial index.
        # radius_neighbors includes the query point itself (dist(p,p)=0 <= eps),
        # which is the count used by the core test.
        nn = NearestNeighbors(radius=self.eps, metric="euclidean").fit(X)
        neighborhoods = nn.radius_neighbors(X, return_distance=False)
        n_neighbors = np.array([len(nb) for nb in neighborhoods])
        is_core = n_neighbors >= self.min_samples

        labels = np.full(n, NOISE, dtype=np.intp)
        cluster_id = 0

        for i in range(n):
            # Only unclassified core points can seed a new cluster.
            if labels[i] != NOISE or not is_core[i]:
                continue

            labels[i] = cluster_id
            stack = [i]
            while stack:
                p = stack.pop()
                if not is_core[p]:
                    continue
                for q in neighborhoods[p]:
                    if labels[q] == NOISE:
                        labels[q] = cluster_id
                        if is_core[q]:
                            stack.append(q)
            cluster_id += 1

        self.labels_ = labels
        self.n_clusters_ = cluster_id
        self.core_sample_indices_ = np.where(is_core)[0]
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


def custom_distance(x, y):
    # Euclidean distance; its eps-ball is the shape of the eps-neighborhood.
    return np.sqrt(np.sum((x - y) ** 2))
```
