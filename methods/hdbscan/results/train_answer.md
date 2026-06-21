The task is to partition an unlabeled point set into clusters that can be dense knots, diffuse clouds, or thin filaments, while leaving genuine background points unassigned as noise. The deeper difficulty is that these structures can sit at very different local densities in the same dataset, so a single global density threshold cannot be right everywhere. DBSCAN fixes one radius `eps` and one count `MinPts`; below that radius points are noise and above it they connect. If `eps` is tuned to keep a diffuse cloud together, it will also bridge dense clusters through sparse gaps; if it is tightened to separate the dense clusters, the cloud dissolves into noise. K-Means, by contrast, assumes convex isotropic groups and forces every point into one of them, so it misses non-convex shapes and has no notion of noise. OPTICS avoids committing to one `eps` by producing a reachability plot, but its reachability distance is asymmetric and extracting a flat clustering from the plot is a separate heuristic, so the result is only approximate and still tends toward a single global cut.

The right object is not one level set but the whole nested family of level sets: Hartigan's density-contour tree. A dense cluster survives at high density levels, a diffuse cluster only at lower levels, and noise sits below every useful level. The challenge is to estimate that tree from the data and then read a single flat partition from it, selecting different clusters at different levels rather than taking one horizontal slice.

The method is HDBSCAN, Hierarchical Density-Based Spatial Clustering of Applications with Noise. It builds the entire density hierarchy in one shot and extracts the flat clustering that maximizes total relative excess of mass. The first step is to estimate local density by the core distance: the distance from each point to its `m_pts`-th nearest neighbor, counting the point itself. The reciprocal of this distance is a K-nearest-neighbor density estimate, and `m_pts` acts only as a smoothing parameter. From the core distances and ordinary pairwise distances, HDBSCAN defines the mutual reachability distance between two points as the maximum of their two core distances and the ordinary distance between them. This is symmetric, and it inflates distances involving sparse-region points, which damps the single-linkage chaining pathology that otherwise merges two dense clusters through one thin bridge.

Next, HDBSCAN notices that running DBSCAN* at every possible radius is the same as running single-linkage hierarchical clustering on the complete graph weighted by mutual reachability distance. Keeping only edges whose weight is below a radius gives exactly the DBSCAN* connectivity at that radius, and single-linkage captures the connected-component evolution as the radius shrinks. Because single-linkage only needs the minimum spanning tree, the whole hierarchy can be obtained from one MST. Conceptually, a self-edge at each vertex weighted by its core distance records the density level at which an isolated point stops being core and becomes noise. The raw hierarchy is then condensed with a minimum cluster size: when a component splits, if two or more pieces are large enough they become true child clusters; if only one piece is large enough the cluster merely shrinks; if no piece is large enough the cluster dies. This removes the bookkeeping noise of fringe points falling out one by one.

Finally, HDBSCAN selects the flat clustering. Each cluster is scored by its relative excess of mass, or stability: for each point in the cluster, add the width of the density interval over which that point belongs to the cluster before it becomes noise or moves into a child, and sum these widths across points. Because raw excess of mass is monotone along a branch and always favors ancestors, the relative version subtracts each cluster's own birth density so parents and children are comparable. The extraction is a bottom-up dynamic program on the condensed tree: at each internal node, either keep the node itself, earning its stability, or discard it and keep the best selections from its children, earning their propagated sum. Ties favor the parent. The selected non-root nodes become the flat clusters, and any point not covered by a selected cluster is labeled noise. This gives one intuitive clustering in which dense and diffuse clusters can coexist.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import pdist, squareform


class HDBSCAN(BaseEstimator, ClusterMixin):
    """Hierarchical density-based clustering with excess-of-mass extraction.
    Noise points get label -1."""

    def __init__(self, min_cluster_size=5, min_samples=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples            # defaults to min_cluster_size
        self.labels_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        m_pts = self.min_samples if self.min_samples is not None else self.min_cluster_size
        m_pts = min(max(m_pts, 1), n - 1)
        mcs = max(self.min_cluster_size, 2)

        D = squareform(pdist(X))
        core = np.partition(D, m_pts, axis=1)[:, m_pts]
        MR = np.maximum(np.maximum(D, core[None, :]), core[:, None])

        mst = self._prim_mst(MR)
        dendro = self._single_linkage(mst, n)
        tree = self._condense(dendro, n, mcs)
        stab = self._stability(tree)
        self.labels_ = self._extract_eom(tree, stab, n)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def predict(self, X):
        if self.labels_ is None:
            self.fit(X)
        return self.labels_

    def _prim_mst(self, MR):
        n = MR.shape[0]
        in_tree = np.zeros(n, dtype=bool)
        best = np.full(n, np.inf)
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
        mst = mst[np.argsort(mst[:, 2], kind="stable")]
        parent = np.arange(2 * n - 1)
        size = np.concatenate([np.ones(n, dtype=np.intp), np.zeros(n - 1, dtype=np.intp)])
        nxt = n

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
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

    def _condense(self, hierarchy, n, mcs):
        children = {n + i: (int(hierarchy[i, 0]), int(hierarchy[i, 1]), hierarchy[i, 2])
                    for i in range(n - 1)}

        node_size = np.ones(2 * n - 1, dtype=np.intp)
        for i in range(n - 1):
            l, r, _ = children[n + i]
            node_size[n + i] = node_size[l] + node_size[r]

        def leaves(node):
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
        rows, ignore, stack = [], set(), [root]
        while stack:
            node = stack.pop()
            if node in ignore or node < n:
                continue
            l, r, dist = children[node]
            lam = (1.0 / dist) if dist > 0 else np.inf
            lc, rc = int(node_size[l]), int(node_size[r])
            if lc >= mcs and rc >= mcs:
                for ch, csz in ((l, lc), (r, rc)):
                    relabel[ch] = nxt
                    rows.append((relabel[node], nxt, lam, csz))
                    nxt += 1
                    stack.append(ch)
            elif lc < mcs and rc < mcs:
                for side in (l, r):
                    for p in leaves(side):
                        rows.append((relabel[node], p, lam, 1))
                    ignore.add(side)
            else:
                big, small = (l, r) if lc >= rc else (r, l)
                relabel[big] = relabel[node]
                stack.append(big)
                for p in leaves(small):
                    rows.append((relabel[node], p, lam, 1))
                ignore.add(small)

        return np.array(rows, dtype=[("parent", np.intp), ("child", np.intp),
                                     ("lambda", np.float64), ("size", np.intp)])

    def _stability(self, tree):
        cluster_ids = set(tree["parent"].tolist())
        births = {}
        for child, lam in zip(tree["child"], tree["lambda"]):
            if child in cluster_ids:
                births[child] = min(births.get(child, np.inf), lam)
        stab = {}
        for parent, lam, sz in zip(tree["parent"], tree["lambda"], tree["size"]):
            b = births.get(parent, 0.0)
            stab[parent] = stab.get(parent, 0.0) + (lam - b) * sz
        return stab

    def _extract_eom(self, tree, stability, n):
        cluster_ids = sorted(stability.keys())
        children = {c: [] for c in cluster_ids}
        for parent, child in zip(tree["parent"], tree["child"]):
            if child in stability:
                children[parent].append(child)

        selected = {c: True for c in cluster_ids}
        prop = dict(stability)
        root = min(cluster_ids)
        for node in sorted(cluster_ids, reverse=True):
            if node == root:
                continue
            sub = sum(prop[c] for c in children[node])
            if sub > stability[node]:
                selected[node] = False
                prop[node] = sub
            else:
                stk = list(children[node])
                while stk:
                    d = stk.pop()
                    selected[d] = False
                    stk.extend(children[d])

        chosen = [c for c in cluster_ids if selected[c] and c != root]
        label_of = {c: i for i, c in enumerate(sorted(chosen))}
        chosen_set = set(chosen)

        parent_of = {child: parent for parent, child in zip(tree["parent"], tree["child"])}
        labels = np.full(n, -1, dtype=np.intp)
        for parent, child in zip(tree["parent"], tree["child"]):
            if child >= n:
                continue
            anc = parent
            while anc is not None and anc not in chosen_set:
                anc = parent_of.get(anc)
            if anc in chosen_set:
                labels[child] = label_of[anc]
        return labels
```
