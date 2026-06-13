Let me start from what actually breaks when I try to cluster real data by density. I have a set of points and I believe there's an unknown density `f` behind them: clusters are the dense regions, noise is the sparse remainder. The cleanest way to make that precise is Hartigan's picture — fix a level `lambda`, look at the level set `{x : f(x) >= lambda}`, and call each of its maximal connected components a cluster. That's exactly right conceptually: noise is just everything below `lambda`, clusters can be any shape because they're connected components, and I don't have to force every point into a group. So if I had `f` I'd be done. I don't, so I estimate it, and the way DBSCAN does this is to fix a radius `eps` and a count `MinPts`: a point is "core" if its `eps`-ball holds at least `MinPts` points, and clusters are connected runs of core points. Fine. But now I'm staring at the real problem: `eps` is a single number, and it sets one global density level. If my data has a tight knot and a diffuse cloud in the same picture, there is no single `eps` that keeps both. Crank `eps` down to resolve the knot and the cloud falls apart into noise; raise it to hold the cloud together and the knot fuses with whatever sparse stuff is near it. One threshold for a structure that genuinely lives at many densities. That's the wall, and it's not a tuning annoyance; it is a structural mismatch between a single level set and a multi-level object.

The obvious escape is "don't pick one `eps` — look at all of them." Vary `lambda`, equivalently vary `eps = 1/lambda`, and watch the components evolve. As I raise the level, a component shrinks by shedding its fringe, or it splits into two, or it disappears. That nested family over all levels is Hartigan's density-contour tree, and it's the honest object: it holds the knot's cluster, which lives at high `lambda`, and the cloud's cluster, which lives at low `lambda`, as different nodes of one tree. So the goal sharpens: build the whole tree of all density-based clusterings at once, and then read a single flat partition off it by taking clusters from different levels rather than one horizontal slice.

How do I build all DBSCAN clusterings for all `eps` without rerunning DBSCAN at a thousand radii? Let me look hard at what DBSCAN's cluster relation actually is. I'll use the cleaner variant that drops border points and keeps clusters as exactly the connected components of core objects, because then a cluster is precisely a chunk of the level set and the interpretation stays clean. Two core objects `p` and `q` end up in the same cluster at radius `eps` when there's a chain of core objects linking them, each consecutive pair within `eps`. The atomic relation is: `p` and `q` are directly reachable at `eps` iff `d(p,q) <= eps`, `p` is core, and `q` is core.

Let me name the radius at which a point first becomes core. The core distance of `p`, `d_core(p)`, is the distance from `p` to its `m_pts`-th nearest neighbor, counting `p` itself as the first neighbor. Then `|N_eps(p)| >= m_pts` exactly when `eps >= d_core(p)`. Its reciprocal `1/d_core(p)` is a K-nearest-neighbor density estimate with `K = m_pts`: small core distance means tightly packed means high density. So `m_pts` is a smoothing factor for the density estimate, not a mysterious cluster-count knob.

Now, when are `p` and `q` directly reachable at radius `eps`? I need `eps >= d_core(p)`, `eps >= d_core(q)`, and `eps >= d(p,q)`. All three at once means

    eps >= max( d_core(p), d_core(q), d(p,q) ).

The smallest radius at which the pair can be directly density-reachable is that maximum, so I define

    d_mreach(p, q) = max( d_core(p), d_core(q), d(p,q) ).

This is a symmetric transformed dissimilarity. It reduces to the original distance whenever the two core distances are no larger than the ordinary pairwise distance, and it inflates a pair when either endpoint lies in a sparse region. Geometrically, a sparse point cannot be closer than its own core distance to anything. That is exactly what I need against single-linkage chaining: sparse bridges become long edges, while dense interiors mostly stay as they were.

OPTICS already points in this direction, but its reachability distance is asymmetric: `reach(o, p) = max(d_core(p), d(p,o))` is measured from `p`, so swapping `p` and `o` can change the value. That asymmetry is why the reachability plot needs heuristic extraction and only approximately corresponds to DBSCAN clusters. Taking the max over both core distances removes that wrinkle. Now I can say the exact thing I wanted: `p` and `q` are directly reachable at radius `eps` if and only if `d_mreach(p,q) <= eps`.

So I build the complete mutual-reachability graph on the data. At radius `eps`, keeping only graph edges with weight `<= eps` gives the same connectivity relation as DBSCAN* at that radius, because the edge condition already enforces that both endpoints are core. Lowering `eps` removes heavier edges first. Removing edges from a weighted graph in decreasing order and tracking connected components is the graph definition of single-linkage hierarchical clustering. So running DBSCAN* over every radius is the same as running single-linkage once, after I replace ordinary distances by mutual reachability distances.

Single-linkage only needs the minimum spanning tree. The MST contains every edge that can ever be the lightest connector between two components, so it carries the whole single-linkage hierarchy. I can compute the MST of the mutual-reachability graph by Prim's algorithm in `O(n^2)` in the dense case, or with spatial-tree variants when the metric allows it, and then sort the MST edges to obtain the agglomerative linkage. Reading that linkage backward is the divisive picture: remove MST edges in decreasing weight order and components split.

There is one event an ordinary MST edge cannot record. If a single object is the last survivor of a shrinking cluster, it still has to become noise once `eps` falls below its own `d_core`. That is not a merge edge between two different objects; it is a one-point density event. The conceptual fix is to extend the MST with a self-edge at each vertex, weighted by that vertex's core distance. Since every incident mutual-reachability edge has weight at least `d_core(v)`, a vertex loses heavier connecting edges before its self-edge, with equal weights handled as ties. The self-edge marks the radius at which the isolated object stops being core and becomes noise. Now the hierarchy encodes both component splits and core-to-noise transitions.

The raw hierarchy is too large to use directly. Most edge removals are not meaningful splits; a fringe point falls out, the cluster is otherwise the same object, and calling that a new cluster is noise in the bookkeeping. I need the density-tree events Hartigan's model actually cares about: shrink, split, die. A minimum cluster size `m_clSize` gives the rule. When one edge, or a tied set of edges, breaks a component into subcomponents, any piece with fewer than `m_clSize` points is spurious. If two or more non-spurious pieces remain, the parent has a true split and each large piece gets a new cluster label. If exactly one non-spurious piece remains, the cluster merely shrinks and that piece keeps the old label while the small pieces become noise. If no non-spurious piece remains, the cluster dies. This condensation turns the raw dendrogram into a compact cluster tree of significant components.

Now I still need a flat clustering. A horizontal cut is exactly what failed at the beginning, so I need to choose clusters from different levels of the condensed tree. The quantity I want is prominence: a cluster should be preferred if many of its points remain members across a wide density interval. Plain lifetime is too crude, because fringe points and core points leave at different levels. Plain excess of mass is also wrong for selection: a parent cluster's excess includes the excesses of its descendants, so raw excess of mass is monotone along a branch and always biases me toward ancestors.

The fix is relative excess of mass. For a cluster `C_i` born at density level `lambda_min(C_i)`, a point `x_j` contributes only for the density interval over which it belongs to this cluster before it becomes noise or moves into a child. Let that departure level be `lambda_max(x_j, C_i)`. With `lambda = 1/eps`, the discrete stability is

    S(C_i) = sum_{x_j in C_i} ( lambda_max(x_j, C_i) - lambda_min(C_i) )
           = sum_{x_j in C_i} ( 1/eps_min(x_j, C_i) - 1/eps_max(C_i) ).

The reciprocal direction matters. The birth radius `eps_max(C_i)` is large because the birth density `lambda_min(C_i)` is low. A point's departure radius `eps_min(x_j, C_i)` is smaller because its departure density `lambda_max(x_j, C_i)` is higher. So each term is nonnegative: higher departure density minus lower birth density. Computationally, the condensed tree already gives me the departure lambda for each child edge, and the birth of a cluster is the minimum lambda at which that cluster appears, with the root's birth set to zero. Accumulating `(lambda_depart - lambda_birth(parent)) * child_size` over the condensed-tree rows gives the stability.

The extraction problem is now clean. I want a non-overlapping set of selected clusters maximizing total stability. I cannot select both a node and one of its descendants, because that would assign a point two labels. On the tree, that becomes a bottom-up dynamic program. For a leaf, the best subtree value is its own stability. For an internal node `C_i`, the choice is either keep `C_i` itself, earning `S(C_i)`, or discard it and keep the best selections already propagated by its children, earning `sum_children S_hat`. Therefore

    S_hat(C_i) = S(C_i)                                      if C_i is a leaf,
    S_hat(C_i) = max(S(C_i), sum_children S_hat(C_child))     otherwise.

The comparison direction is easy to get backwards, so I check it directly. If the children's propagated sum is strictly greater than the node's own stability, the finer structure is more stable, so I deselect the node and propagate the children's sum upward. If the node's stability is greater or equal, the parent is the better representative, so I select it and suppress the whole subtree below it. That tie direction selects the parent, exactly matching the usual EOM rule. Excluding the root as a valid cluster gives the flat clustering; points not covered by any selected cluster are noise.

The same hierarchy also gives useful side products. For an outlier score, the closest density-based cluster of a point is the first cluster it attaches to when I lower the density threshold. Compare the point's own radius `eps(x_i)` to the smallest radius reached by the densest member of that reference cluster, `eps_max(x_i)`, and the score `1 - eps_max(x_i)/eps(x_i)` is near zero for core-like members and near one for loosely attached points. For semi-supervised extraction, I can replace each cluster's stability objective with a constraint-satisfaction score and run the same tree DP. The machinery is the same because the expensive part is the hierarchy.

When I turn this into code, I have to keep the neighbor-count convention straight. The mathematical `m_pts` here counts the point itself in its `m_pts`-th-nearest-neighbor radius (so does `sklearn.cluster.HDBSCAN`'s `min_samples`); `scikit-learn-contrib/hdbscan` exposes a `min_samples` value that does *not* count the query point, so to reproduce the same core distances there I'd use a value one smaller. I'll fill the estimator's single empty slot with the pipeline I derived — core distances by partial sort, mutual reachability by two elementwise maxes, MST by Prim's, single-linkage by union-find, the size-based condensation, the relative-excess-of-mass stability, and the bottom-up EOM dynamic program — computing subtree sizes bottom-up and walking the tree iteratively so a deep hierarchy doesn't overflow the recursion stack.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import pdist, squareform


class CustomClustering(BaseEstimator, ClusterMixin):
    """HDBSCAN: single-linkage hierarchy in mutual-reachability space, condensed
    by a minimum cluster size, then the flat clustering that maximizes total
    relative excess of mass (excess-of-mass selection). Noise gets label -1."""

    def __init__(self, n_clusters=None, random_state=42,
                 min_cluster_size=5, min_samples=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.min_cluster_size = min_cluster_size
        # min_samples is m_pts (density smoothing); defaults to min_cluster_size
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        m_pts = self.min_samples if self.min_samples is not None else self.min_cluster_size
        m_pts = min(max(m_pts, 1), n - 1)
        mcs = max(self.min_cluster_size, 2)

        D = squareform(pdist(X))                       # pairwise distances d(p,q)
        # core distance: distance to the m_pts-th NN (incl. self). 1/core is the
        # K-NN density estimate with K = m_pts.
        core = np.partition(D, m_pts, axis=1)[:, m_pts]
        # mutual reachability max(d_core(p), d_core(q), d(p,q)): symmetric, and
        # inflates sparse-region distances so single-linkage chaining is damped.
        MR = np.maximum(np.maximum(D, core[None, :]), core[:, None])

        mst = self._prim_mst(MR)                        # MST of the mreach graph
        dendro = self._single_linkage(mst, n)          # remove edges desc -> dendrogram
        condensed = self._condense(dendro, n, mcs)     # true split vs shrink vs die
        stability = self._stability(condensed)         # relative excess of mass
        self.labels_ = self._extract_eom(condensed, stability, n)
        return self

    def predict(self, X):
        if self.labels_ is None:
            self.fit(X)
        return self.labels_

    # --- single-linkage in mutual-reachability space -----------------------

    def _prim_mst(self, MR):
        # Prim's: grow a tree from node 0, each step add the cheapest edge to a
        # node not yet in the tree. Returns (n-1, 3): src, dst, weight.
        n = MR.shape[0]
        in_tree = np.zeros(n, dtype=bool)
        best = np.full(n, np.inf)                       # cheapest known edge into each node
        src = np.zeros(n, dtype=np.intp)
        edges = np.zeros((n - 1, 3))
        cur = 0
        for i in range(n - 1):
            in_tree[cur] = True
            d = MR[cur]
            upd = (~in_tree) & (d < best)
            best[upd] = d[upd]
            src[upd] = cur
            best[in_tree] = np.inf
            nxt = int(np.argmin(best))
            edges[i] = (src[nxt], nxt, best[nxt])
            cur = nxt
        return edges

    def _single_linkage(self, mst, n):
        # Union MST edges in ascending order (agglomerative); reverse = removing
        # them descending splits components. Rows: [left, right, dist, size].
        mst = mst[np.argsort(mst[:, 2], kind="stable")]
        parent = np.arange(2 * n - 1)                   # union-find over 2n-1 nodes
        size = np.concatenate([np.ones(n, dtype=np.intp),
                               np.zeros(n - 1, dtype=np.intp)])
        nxt = n

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:                    # path compression
                parent[x], x = root, parent[x]
            return root

        out = np.zeros((n - 1, 4))
        for i in range(n - 1):
            a, b, dist = int(mst[i, 0]), int(mst[i, 1]), mst[i, 2]
            ra, rb = find(a), find(b)
            out[i] = (ra, rb, dist, size[ra] + size[rb])
            parent[ra] = parent[rb] = nxt
            size[nxt] = size[ra] + size[rb]
            nxt += 1
        return out

    # --- condense the tree with a minimum cluster size ---------------------

    def _condense(self, hierarchy, n, mcs):
        # Walk the dendrogram top-down (a merge read in reverse = a split as the
        # threshold drops). With lambda = 1/dist as the density level:
        #   both children >= mcs  -> TRUE split: two new clusters
        #   both children <  mcs  -> cluster dies: all points fall out as noise
        #   one child   <  mcs    -> cluster SHRINKS: big child keeps the label,
        #                            small child's points fall out as noise here
        children = {n + i: (int(hierarchy[i, 0]), int(hierarchy[i, 1]), hierarchy[i, 2])
                    for i in range(n - 1)}

        # subtree sizes bottom-up (merge i forms node n+i from its two children)
        node_size = np.ones(2 * n - 1, dtype=np.intp)
        for i in range(n - 1):
            l, r, _ = children[n + i]
            node_size[n + i] = node_size[l] + node_size[r]

        def leaves(node):                               # iterative; no deep recursion
            if node < n:
                return [node]
            out, stack = [], [node]
            while stack:
                cur = stack.pop()
                if cur < n:
                    out.append(cur)
                else:
                    l, r, _ = children[cur]
                    stack.append(l)
                    stack.append(r)
            return out

        root = 2 * (n - 1)
        relabel = {root: n}
        nxt = n + 1
        rows, ignore, stack = [], set(), [root]         # (parent, child, lambda, size)
        while stack:
            node = stack.pop()
            if node in ignore or node < n:
                continue
            l, r, dist = children[node]
            lam = (1.0 / dist) if dist > 0 else np.inf
            lc, rc = int(node_size[l]), int(node_size[r])

            if lc >= mcs and rc >= mcs:                 # TRUE split
                for ch, csz in ((l, lc), (r, rc)):
                    relabel[ch] = nxt
                    rows.append((relabel[node], nxt, lam, csz))
                    nxt += 1
                    stack.append(ch)
            elif lc < mcs and rc < mcs:                 # cluster disappears
                for side in (l, r):
                    for p in leaves(side):
                        rows.append((relabel[node], p, lam, 1))
                    ignore.add(side)
            else:                                       # cluster shrinks
                big, small = (l, r) if lc >= rc else (r, l)
                relabel[big] = relabel[node]            # survivor keeps the label
                stack.append(big)
                for p in leaves(small):                 # small side -> noise here
                    rows.append((relabel[node], p, lam, 1))
                ignore.add(small)

        return np.array(rows, dtype=[("parent", np.intp), ("child", np.intp),
                                     ("lambda", np.float64), ("size", np.intp)])

    # --- relative excess of mass (cluster stability) -----------------------

    def _stability(self, condensed):
        # birth(C) = lambda at which cluster C first appears (min lambda of its
        # incoming edges; root birth = 0). stability(C) = sum over departures of
        # (lambda_depart - birth(C)) * count = relative excess of mass.
        cluster_ids = set(condensed["parent"].tolist())
        births = {}
        for child, lam in zip(condensed["child"], condensed["lambda"]):
            if child in cluster_ids:                    # child is itself a cluster node
                births[child] = min(births.get(child, np.inf), lam)
        stab = {}
        for parent, lam, sz in zip(condensed["parent"], condensed["lambda"],
                                   condensed["size"]):
            b = births.get(parent, 0.0)
            stab[parent] = stab.get(parent, 0.0) + (lam - b) * sz
        return stab

    # --- excess-of-mass extraction (bottom-up dynamic program) -------------

    def _extract_eom(self, condensed, stability, n):
        cluster_ids = sorted(stability.keys())
        children = {c: [] for c in cluster_ids}
        for parent, child in zip(condensed["parent"], condensed["child"]):
            if child in stability:                      # child is a cluster, not a point
                children[parent].append(child)

        selected = {c: True for c in cluster_ids}
        prop = dict(stability)                          # S_hat propagated up the tree
        root = min(cluster_ids)
        for node in sorted(cluster_ids, reverse=True):  # deepest (largest id) first
            if node == root:                            # root is not a valid cluster
                continue
            sub = sum(prop[c] for c in children[node])
            if sub > stability[node]:                   # children strictly better: keep them
                selected[node] = False
                prop[node] = sub
            else:                                       # node wins (ties -> parent); collapse subtree
                stk = list(children[node])
                while stk:
                    d = stk.pop()
                    selected[d] = False
                    stk.extend(children[d])

        chosen = [c for c in cluster_ids if selected[c] and c != root]
        label_of = {c: i for i, c in enumerate(sorted(chosen))}
        chosen_set = set(chosen)

        # assign each point to the selected cluster on its root-to-leaf path
        parent_of = {child: parent for parent, child in
                     zip(condensed["parent"], condensed["child"])}
        labels = np.full(n, -1, dtype=np.intp)          # -1 = noise
        for parent, child in zip(condensed["parent"], condensed["child"]):
            if child >= n:                              # only points carry final labels
                continue
            anc = parent
            while anc is not None and anc not in chosen_set:
                anc = parent_of.get(anc)
            if anc in chosen_set:
                labels[child] = label_of[anc]
        return labels
```

The causal chain is now clear: I start from the failure of one global density threshold, replace the radius choice with the full density hierarchy, notice that DBSCAN* direct reachability is exactly mutual reachability `<= eps`, and get all radii at once by single-linkage on the mutual-reachability MST. I add core-distance self-edges conceptually so isolated points have a core-to-noise level, condense the raw hierarchy by distinguishing true splits from shrinkage and death, score each surviving cluster by the relative excess-of-mass span `lambda_max - lambda_min`, and select the flat clustering by the bottom-up EOM dynamic program. The result keeps arbitrary shapes and noise, but the selected clusters can live at different density levels, so one diffuse cluster and one dense cluster can both appear in the same final labeling.
