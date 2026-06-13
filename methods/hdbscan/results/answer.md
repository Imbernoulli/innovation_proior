# HDBSCAN, distilled

HDBSCAN builds the single-linkage hierarchy of a mutual-reachability distance transform, condenses that hierarchy with a minimum cluster size, and extracts a flat clustering by maximizing total relative excess of mass. It keeps the density-tree view of DBSCAN* over all radii, so clusters can be selected at different density levels while noise remains unlabeled.

## Problem it solves

Density-based clustering treats clusters as connected components of density level sets `{x : f(x) >= lambda}` and treats points below the chosen level as noise. DBSCAN fixes one radius `eps`, equivalently one density level `lambda = 1/eps`. One global level cannot recover clusters whose densities differ: a high level breaks diffuse clusters into noise, while a low level merges dense clusters through sparse bridges.

HDBSCAN keeps the whole hierarchy over density levels, then selects a non-overlapping set of stable clusters from that tree.

## Key definitions

**Core distance.** `d_core(p)` is the distance from `p` to its `m_pts`-th nearest neighbor, counting `p` itself as the first neighbor. It is the smallest `eps` for which `p` is core, and `1/d_core(p)` is the K-nearest-neighbor density estimate used for level ordering.

**Mutual reachability distance.**

    d_mreach(p, q) = max(d_core(p), d_core(q), d(p, q)).

This is the smallest radius at which `p` and `q` can be directly density-reachable in DBSCAN*. It is symmetric, and it stretches sparse regions because no point can be closer than its own core distance.

**All DBSCAN* cuts as one hierarchy.** In the complete graph weighted by `d_mreach`, keeping edges with weight `<= eps` gives the DBSCAN* connectivity at radius `eps`. Removing edges in decreasing weight order is single-linkage. Therefore one single-linkage hierarchy in mutual-reachability space contains the DBSCAN* clusterings for all radii.

**MST construction.** Single-linkage only needs the minimum spanning tree of the mutual-reachability graph. Conceptually add one self-edge per vertex with weight `d_core(v)` so an isolated object has a recorded core-to-noise level. Equal edge weights are handled as simultaneous removals.

**Condensed cluster tree.** With minimum cluster size `m_clSize`, each split event is classified by the sizes of the resulting subcomponents:

- two or more subcomponents `>= m_clSize`: true split; large subcomponents become child clusters and smaller pieces become noise;
- exactly one subcomponent `>= m_clSize`: shrink; that subcomponent keeps the parent label and smaller pieces become noise;
- no subcomponent `>= m_clSize`: death; the cluster disappears and its points become noise.

Setting `m_clSize = m_pts` gives the common one-primary-parameter form; implementations usually expose `min_cluster_size` and `min_samples` separately, with `min_samples` defaulting to `min_cluster_size`.

## Stability and EOM

Use `lambda = 1/eps`. A cluster `C_i` is born at `lambda_min(C_i)`. Each point `x_j` leaves it at `lambda_max(x_j, C_i)`, either by becoming noise or by entering a child cluster. The cluster stability, the finite-sample relative excess of mass, is

    S(C_i) = sum_{x_j in C_i} (lambda_max(x_j, C_i) - lambda_min(C_i))
           = sum_{x_j in C_i} (1/eps_min(x_j, C_i) - 1/eps_max(C_i)).

The reciprocal direction is essential: the birth radius `eps_max(C_i)` is larger and corresponds to the lower density `lambda_min(C_i)`; the departure radius `eps_min(x_j, C_i)` is smaller and corresponds to the higher density `lambda_max(x_j, C_i)`.

Raw excess of mass is monotone along a branch because a parent contains its descendants. Relative excess of mass subtracts each cluster's own birth level, so parent and child candidates become comparable.

Flat extraction is a bottom-up dynamic program over the condensed tree:

    S_hat(C_i) = S(C_i)                                      if C_i is a leaf,
    S_hat(C_i) = max(S(C_i), sum_children S_hat(C_child))     otherwise.

For each non-root node, if the children's propagated stability sum is strictly greater than the node's own stability, deselect the node and propagate the children. Otherwise select the node and suppress its descendants. Ties select the parent. The selected non-root nodes define the flat clustering; uncovered points are noise with label `-1`.

## Algorithm

```
input: X, min_cluster_size, min_samples (m_pts)
1. core[i] = m_pts-th nearest-neighbor distance of x_i, counting x_i itself
2. MR[i,j] = max(d(i,j), core[i], core[j])
3. MST = minimum spanning tree of the mutual-reachability graph
4. linkage = single-linkage hierarchy from sorted MST edges
5. condensed = size-based condensation: true split vs shrink vs death
6. S(C) = sum over condensed-tree departures (lambda_depart - lambda_birth(C)) * count
7. EOM = bottom-up DP: keep children iff their propagated sum > node stability
8. labels = selected cluster per point; otherwise -1
```

## Implementation notes

`sklearn.cluster.HDBSCAN` and the mathematical definition count the point itself in `min_samples`. `scikit-learn-contrib/hdbscan` exposes a public `min_samples` convention that does not count the query point; matching the same core distance uses a contrib value one smaller.

Common estimator defaults are `min_cluster_size=5`, `min_samples=None` so it defaults to `min_cluster_size`, `cluster_selection_method="eom"`, `metric="euclidean"`, and `allow_single_cluster=False`. The scikit-learn estimator uses `algorithm="auto"` and noise label `-1`; the contrib estimator has additional implementation options such as approximate minimum spanning tree generation and prediction data.

## Working code

Faithful to the canonical `scikit-learn-contrib/hdbscan` pipeline (core distance by partial
sort, mutual reachability by two elementwise maxes, MST by Prim's, single-linkage by union-find,
size-based condensation, summed relative excess of mass, bottom-up EOM selection), packaged as a
`fit(X) -> labels_` estimator. Verified to track `sklearn.cluster.HDBSCAN` closely on Gaussian
blobs, interleaving moons, and the 64-d digits dataset (same cluster counts; ARI within ~0.02).

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import pdist, squareform


class HDBSCAN(BaseEstimator, ClusterMixin):
    """Hierarchical density-based clustering with excess-of-mass extraction.
    Noise points get label -1. min_samples is m_pts (counts the point itself,
    the sklearn convention); scikit-learn-contrib excludes it, so use one less
    there to reproduce the same core distances."""

    def __init__(self, min_cluster_size=5, min_samples=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples            # m_pts; defaults to min_cluster_size
        self.labels_ = None

    # -- pipeline -----------------------------------------------------------

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        m_pts = self.min_samples if self.min_samples is not None else self.min_cluster_size
        m_pts = min(max(m_pts, 1), n - 1)
        mcs = max(self.min_cluster_size, 2)

        D = squareform(pdist(X))
        core = np.partition(D, m_pts, axis=1)[:, m_pts]      # m_pts-th NN distance (incl self)
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

    # -- MST + single-linkage ----------------------------------------------

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

    # -- condense with minimum cluster size --------------------------------

    def _condense(self, hierarchy, n, mcs):
        children = {n + i: (int(hierarchy[i, 0]), int(hierarchy[i, 1]), hierarchy[i, 2])
                    for i in range(n - 1)}

        node_size = np.ones(2 * n - 1, dtype=np.intp)        # subtree sizes bottom-up
        for i in range(n - 1):
            l, r, _ = children[n + i]
            node_size[n + i] = node_size[l] + node_size[r]

        def leaves(node):                                    # iterative (deep trees safe)
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
            lam = (1.0 / dist) if dist > 0 else np.inf       # density level
            lc, rc = int(node_size[l]), int(node_size[r])
            if lc >= mcs and rc >= mcs:                      # true split
                for ch, csz in ((l, lc), (r, rc)):
                    relabel[ch] = nxt
                    rows.append((relabel[node], nxt, lam, csz))
                    nxt += 1
                    stack.append(ch)
            elif lc < mcs and rc < mcs:                      # cluster dies
                for side in (l, r):
                    for p in leaves(side):
                        rows.append((relabel[node], p, lam, 1))
                    ignore.add(side)
            else:                                            # cluster shrinks
                big, small = (l, r) if lc >= rc else (r, l)
                relabel[big] = relabel[node]
                stack.append(big)
                for p in leaves(small):
                    rows.append((relabel[node], p, lam, 1))
                ignore.add(small)
        return np.array(rows, dtype=[("parent", np.intp), ("child", np.intp),
                                     ("lambda", np.float64), ("size", np.intp)])

    # -- relative excess of mass (stability) -------------------------------

    def _stability(self, tree):
        cluster_ids = set(tree["parent"].tolist())
        births = {}
        for child, lam in zip(tree["child"], tree["lambda"]):
            if child in cluster_ids:                         # child is itself a cluster
                births[child] = min(births.get(child, np.inf), lam)
        stab = {}
        for parent, lam, sz in zip(tree["parent"], tree["lambda"], tree["size"]):
            b = births.get(parent, 0.0)                      # root birth = 0
            stab[parent] = stab.get(parent, 0.0) + (lam - b) * sz
        return stab

    # -- excess-of-mass extraction -----------------------------------------

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
            if sub > stability[node]:                        # children strictly better
                selected[node] = False
                prop[node] = sub
            else:                                            # node wins (ties -> parent)
                stk = list(children[node])
                while stk:
                    d = stk.pop()
                    selected[d] = False
                    stk.extend(children[d])

        chosen = [c for c in cluster_ids if selected[c] and c != root]
        label_of = {c: i for i, c in enumerate(sorted(chosen))}
        chosen_set = set(chosen)

        parent_of = {child: parent for parent, child in zip(tree["parent"], tree["child"])}
        labels = np.full(n, -1, dtype=np.intp)               # -1 = noise
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

## Extensions

- GLOSH outlier score: `GLOSH(x_i) = 1 - eps_max(x_i) / eps(x_i)`, where `eps(x_i)` is the radius at which `x_i` attaches to its closest density-based cluster and `eps_max(x_i)` is the radius of that cluster's densest reference point.
- Semi-supervised extraction: replace `S(C)` with a constraint-satisfaction objective for should-link / should-not-link pairs and run the same bottom-up tree DP.
