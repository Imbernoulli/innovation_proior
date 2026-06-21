On almost every structured-data problem worth fighting over — insurance claims, ad click logs, web-search ranking, physics event classification — the model that wins is a boosted ensemble of regression trees. Accuracy is not the issue. The issue is that the data has outgrown the tools: click logs and claims tables reach hundreds of millions to billions of rows, the feature matrices are mostly empty (missing values, frequent zeros, one-hot encodings that turn one categorical into a thousand sparse columns), and the boosting packages everyone uses assume the data sits in RAM, re-sort it at every node of every tree, and special-case sparsity with whatever ad-hoc hack the user happened to wire up. The goal, then, is not a cleverer loss but one end-to-end system that is *both* as accurate as the best tree boosting *and* able to push a terabyte of sparse data through a desktop or a small cluster — where the accuracy and the scale come from the same design rather than being bolted together.

The standard recipe leaves an opening, and the opening is the same place where both the accuracy and the scaling improvements live. Friedman's gradient boosting fits the ensemble $\hat y_i = \sum_{k=1}^K f_k(x_i)$ — each $f_k(x) = w_{q(x)}$ a regression tree, $q$ routing an example to one of $T$ leaves and $w \in \mathbb{R}^T$ the leaf scores — by steepest descent in function space: fit a tree by least squares to the negative-gradient pseudo-residuals to choose its *shape*, then a separate per-leaf line search on the real loss to choose its *values*. That two-step structure is quietly incoherent. The criterion that picks the splits (least-squares fit to the gradient) is not the quantity actually being minimized; for squared error they coincide, so nobody worries, but in general the split-selection criterion and the objective have drifted apart. Worse, there is no model-complexity term *inside* the criterion that picks splits — the only regularization is shrinkage and early stopping, both applied outside the tree-growing decision. So when the algorithm asks "is this split worth it?", it answers with no notion of the cost of adding a leaf. The clue to fixing both at once is in Friedman, Hastie and Tibshirani's reading of boosting as adaptive *Newton* stepping: LogitBoost fits each round by weighted least squares to a working response $z_i = (y_i^*-p_i)/(p_i(1-p_i))$ with weights $w_i = p_i(1-p_i)$ — the working response is exactly negative-gradient-over-curvature and the weight is exactly the loss curvature. The second derivative is already implicit in boosting; mainstream gradient boosting throws it away, grows the tree from the bare gradient, and patches the curvature back as a per-leaf afterthought.

I propose XGBoost: a scalable gradient tree boosting system built on a single regularized, second-order objective, in which the same expression that scores a split also determines the leaf values and the same constant that penalizes complexity also prunes. Write the whole-ensemble objective as $$L(\phi) = \sum_i l(\hat y_i, y_i) + \sum_k \Omega(f_k), \qquad \Omega(f) = \gamma T + \tfrac{1}{2}\lambda \lVert w \rVert^2,$$ where $l$ is any differentiable convex loss. The complexity penalty $\Omega$ charges $\gamma$ per leaf — a direct knob on tree size that, as it turns out, doubles as a pruning threshold — and $\tfrac{1}{2}\lambda\lVert w\rVert^2$ is an L2 penalty on the leaf scores, the ridge impulse: do not let any single leaf make a huge, overconfident jump. The factor $\tfrac{1}{2}$ keeps the algebra clean under differentiation, and when $\lambda = 0$, $\gamma = 0$ the objective collapses back to ordinary gradient tree boosting, which is exactly the sanity check I want — this generalizes, it does not replace. (Regularized Greedy Forest also penalizes the forest, but it re-optimizes all leaf weights jointly and fully-correctively, which is the part that is painful to parallelize; I want a penalty I can evaluate *locally* while growing one tree.)

Because the parameters are functions, fit additively: at round $t$, hold $F_{t-1}$ fixed and add the $f_t$ minimizing $\sum_i l(y_i, \hat y_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$. Taylor-expand the loss to second order in the new increment around $\hat y_i^{(t-1)}$, and drop the constant term $l(y_i, \hat y_i^{(t-1)})$ that does not depend on $f_t$: $$\tilde L^{(t)} = \sum_{i=1}^n \big[ g_i\, f_t(x_i) + \tfrac{1}{2} h_i\, f_t^2(x_i) \big] + \Omega(f_t), \qquad g_i = \partial_{\hat y}\, l(y_i, \hat y^{(t-1)}),\ \ h_i = \partial^2_{\hat y}\, l(y_i, \hat y^{(t-1)}).$$ The loss now reaches the tree builder *only* through the pair $(g_i, h_i)$. Squared error gives $g_i = \hat y - y$, $h_i = 1$; logistic gives $g_i = p - y$, $h_i = p(1-p)$; a ranking objective gives its own pair. One split-finding engine therefore serves regression, classification, ranking, and arbitrary user objectives — the loss is decoupled from the learner, which is precisely what lets one *system* become the consensus tool instead of a different package per task.

Now solve the round in closed form. A tree partitions instances by leaf, so with $I_j = \{i : q(x_i) = j\}$, $G_j = \sum_{i\in I_j} g_i$, $H_j = \sum_{i\in I_j} h_i$, the objective becomes a sum of independent per-leaf convex quadratics $G_j w_j + \tfrac{1}{2}(H_j + \lambda) w_j^2 + \gamma T$. Each quadratic $aw + \tfrac{1}{2}bw^2$ with $b = H_j + \lambda > 0$ is minimized at $w^* = -a/b$, so the optimal leaf weight is $$w_j^* = -\frac{G_j}{H_j + \lambda}.$$ This is exactly the Newton step — negative gradient sum over curvature sum — with one new ingredient, the $+\lambda$ in the denominator, and that ingredient is load-bearing: a leaf with tiny curvature $H_j$ (few instances, or a flat region of the loss) would otherwise get a weight $-G_j/H_j$ that explodes; the $+\lambda$ is the Tikhonov term that caps the leaf score and shrinks every leaf toward zero. Substituting $w_j^*$ back, a quadratic at its minimum has value $-\tfrac{1}{2}a^2/b$, so the *structure score* of a fixed tree $q$ is $$\tilde L^{(t)}(q) = -\frac{1}{2}\sum_{j=1}^T \frac{G_j^2}{H_j + \lambda} + \gamma T.$$ This is the single object the old recipe was missing: it scores a partition (playing the role impurity plays in CART, but derived from an arbitrary loss and with complexity charged in) *and* it determines the leaf values, from one expression. Since all structures cannot be enumerated, grow greedily, but score each candidate split with this derived score. Splitting a node into $I_L, I_R$ adds one leaf (cost $+\gamma$) and changes the score, so the loss reduction — the gain, larger is better — is $$\text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma.$$ The fraction terms are the curvature-weighted purity improvement; the $-\gamma$ is the price of the extra leaf. If the best gain is not positive, the split did not pay for itself, so do not make it — pre-pruning that falls straight out of the *same* $\gamma$ that sits in $\Omega$. One constant simultaneously penalizes leaves in the objective and acts as the minimum-gain split threshold; no separate stopping rule is invented.

The inner loop is then: for each feature, sort the node's instances by feature value, sweep left to right maintaining running $G_L, H_L$ (with $G_R = G - G_L$, $H_R = H - H_L$ free), and at each candidate threshold evaluate the gain and track the max. That exact greedy scan considers every split point, but "sort the node's instances per feature" is precisely the operation that does not survive at scale — it cannot be done out of memory, and the whole dataset cannot be re-sorted per node in a distributed setting. So I relax it to an *approximate* finder: propose a small set of candidate split points per feature, bucket the instances between consecutive candidates, aggregate $(G,H)$ per bucket, and search only the bucket boundaries — globally per tree (fewer proposals, more candidates needed) or locally per split (more work, candidates refined for deep trees). The subtlety is *which* candidates. Completing the square in the per-round objective, $g_i f + \tfrac{1}{2}h_i f^2 = \tfrac{1}{2}h_i(f + g_i/h_i)^2 - \tfrac{1}{2}g_i^2/h_i$, reveals it to be *weighted squared error*: $$\tilde L^{(t)} = \sum_i \tfrac{1}{2} h_i\,\big(f_t(x_i) - (-g_i/h_i)\big)^2 + \Omega(f_t) + \text{const},$$ each instance carrying target $-g_i/h_i$ and weight $h_i$. The importance of instance $i$ to this round is its curvature $h_i$. So the candidate splits must be spaced evenly in the *$h$-weighted* rank of the feature, $r_k(z) = \big(\sum_{x<z} h\big)/\big(\sum h\big)$, with $|r_k(s_j) - r_k(s_{j+1})| < \epsilon$ giving $\approx 1/\epsilon$ candidates — and when all $h_i$ are equal (squared error) this reduces to ordinary quantiles, the clean special case.

Computing exact weighted quantiles over a billion distributed rows by sorting is infeasible, and the streaming-quantile sketches the database community offers carry their $\epsilon$-guarantee only for *unweighted* points. So XGBoost includes a weighted quantile sketch with provable $\epsilon$-bounds. A summary $Q(D) = (S, \tilde r^+, \tilde r^-, \tilde\omega)$ over weighted data is $\epsilon$-approximate when $\tilde r^+(y) - \tilde r^-(y) - \tilde\omega(y) \le \epsilon\,\omega(D)$ for every $y$, which pins either rank to within $\epsilon\,\omega(D)$. Two operations preserve this. *Merge* adds the extended rank functions pointwise; because the true ranks are additive over the union and $\omega(D) = \omega(D_1) + \omega(D_2)$, the gap quantity adds and the merged error is $\max(\epsilon_1, \epsilon_2)$ — it does *not* accumulate, which is what makes hierarchical and distributed aggregation work. *Prune* queries the summary at $b+1$ evenly spaced ranks via a midpoint query whose returned point brackets the target rank to within $(\epsilon/2)\omega(D)$, yielding a summary of at most $b+1$ points with error $\epsilon + 1/b$. Merges keep error flat, prunes add only $1/b$, so a streaming/distributed pipeline keeps total error at a controllable $\epsilon$. The same shape of guarantee as the classical Greenwald–Khanna sketch, now for arbitrary weights — and the weights it carries are exactly the curvatures $h_i$ from the second-order objective, so the approximate split finder and the regularized core are the same idea seen twice.

Sparsity is handled not by guessing but by *learning*. Each split node carries a default direction; any instance whose split feature is missing or absent in the sparse representation flows the default way, and the default is chosen from the same gain criterion. For a feature, only the non-missing entries have a value to sort by; the missing block has aggregate $(G,H)$ equal to the node totals minus the present totals. Scan the non-missing entries twice — ascending with the missing block assigned to the right child, then descending with it assigned to the left — and keep whichever sweep gives the higher gain, that sweep's side becoming the learned default. Because both sweeps touch only the non-missing entries, the cost of split finding is *linear in the number of non-zeros*, not in the dense matrix size: sparsity stops being a problem to mitigate and becomes a speedup. Two cheap, well-known regularizers sit on top of the per-tree core without touching any derivation: shrinkage $F_t = F_{t-1} + \eta f_t$ with $\eta \in (0,1)$, which reduces the influence of any single tree and trades off against the number of trees; and column subsampling, the RandomForest trick of considering only a random subset of features per tree, which decorrelates the trees, often curbs overfitting more than row subsampling, and makes split finding cheaper.

The scaling story closes because the structure score needs only *aggregated* $(G,H)$ over instance sets, and aggregation bucketizes and parallelizes cleanly. The one expensive operation is getting data into feature-sorted order, so do not re-sort per node: store the data once in compressed-column (CSC) blocks with each column pre-sorted by feature value, and reuse that layout across every iteration and every node. A single linear scan over a sorted column then enumerates split candidates for all leaves at once — the per-scan $\log n$ factor is paid once and amortized forever. For exact single-machine training the dataset is one block; for approximate, out-of-core, or distributed training, use many blocks, one per row-subset, sharded across machines or spilled to disk, and the weighted-quantile candidate search becomes a linear scan over the sorted columns. The block layout creates one new hazard: scanning a column in feature order fetches the gradient statistics $(g_i, h_i)$ by row index in non-contiguous order, missing the CPU cache. Decouple the fetch from the accumulate — each thread prefetches a batch of $(g_i, h_i)$ into a small internal buffer and accumulates from the buffer in a tight loop, with a block size of about $2^{16}$ instances balancing parallelism against cache residency. For truly out-of-core data: disk-resident blocks with an independent prefetch thread overlapping IO with compute, per-column compression decompressed on the fly, and sharding across disks. None of this touches the math; it all serves the same $(G,H)$-aggregation engine. As a final consistency check, the squared-error special case $l = \tfrac{1}{2}(y-\hat y)^2$ gives $g_i = \hat y_i - y_i$, $h_i = 1$, so $w_j^* = -(\sum_{i\in I_j}(\hat y_i - y_i))/(n_j + \lambda)$ is the regularized mean residual (LS-TreeBoost when $\lambda = 0$), the weighted sketch reduces to ordinary quantiles, and the structure score becomes the classical first-order picture — confirming the construction contains gradient boosting and adds genuine second-order leaf values and split scores for non-quadratic losses, a complexity penalty that doubles as a prune threshold, and the weighted-quantile plus sparsity-aware plus block machinery that lets it run at scale.

The base learner that fills the boosting harness is a regularized second-order regression tree, grown by the derived gain, with leaf weight $-G/(H+\lambda)$, the learned missing-value default direction, and the loss reaching the tree only through $(g,h)$:

```python
import numpy as np


def _leaf_score(G, H, lam):
    # min_w [ G w + 0.5 (H+lam) w^2 ] = -0.5 * G^2 / (H+lam); we return G^2/(H+lam).
    return G * G / (H + lam)


class _Node:
    __slots__ = ("feat", "thresh", "default_left", "left", "right", "weight")

    def __init__(self):
        self.feat = -1            # split feature (-1 => leaf)
        self.thresh = 0.0
        self.default_left = True  # learned direction for missing values
        self.left = self.right = None
        self.weight = 0.0         # leaf score w* (used only at a leaf)


class RegularizedTree:
    """A second-order regression tree grown on (g, h).
    Gain = 0.5[G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam)] - gamma; keep iff Gain > 0
    (gamma is the prune threshold). Missing values (NaN) take a learned default direction."""

    def __init__(self, max_depth=6, lam=1.0, gamma=0.0, min_child_h=1.0):
        self.max_depth, self.lam, self.gamma, self.min_child_h = max_depth, lam, gamma, min_child_h

    def fit(self, X, g, h):
        self.root = self._build(X, g, h, np.arange(len(g)), 0)
        return self

    def _build(self, X, g, h, idx, depth):
        node = _Node()
        G, H = g[idx].sum(), h[idx].sum()
        node.weight = -G / (H + self.lam)                     # optimal leaf weight
        if depth >= self.max_depth or len(idx) <= 1:
            return node
        best = self._best_split(X, g, h, idx, G, H)
        if best is None:                                      # no Gain > 0 => prune to a leaf
            return node
        node.feat, node.thresh, node.default_left, li, ri = best
        node.weight = 0.0
        node.left = self._build(X, g, h, li, depth + 1)
        node.right = self._build(X, g, h, ri, depth + 1)
        return node

    def _best_split(self, X, g, h, idx, G, H):
        lam, parent = self.lam, _leaf_score(G, H, self.lam)
        best_gain, best = 0.0, None                           # must clear 0 (i.e. clear gamma)
        for feat in range(X.shape[1]):                        # column subsampling restricts this
            col = X[idx, feat]
            present = ~np.isnan(col)                           # sparsity-aware: non-missing only
            pid = idx[present]
            mid = idx[~present]                                # rows whose split feature is missing
            if len(pid) < 2:
                continue
            order = np.argsort(col[present], kind="stable")
            pid = pid[order]
            vals = col[present][order]
            csg, csh = np.cumsum(g[pid]), np.cumsum(h[pid])
            Gp, Hp = csg[-1], csh[-1]                          # totals over present rows
            for s in range(1, len(pid)):
                if vals[s] == vals[s - 1]:
                    continue
                thr = 0.5 * (vals[s] + vals[s - 1])
                # Pass A -- missing -> right (left = present prefix)
                gl, hl = csg[s - 1], csh[s - 1]
                gr, hr = G - gl, H - hl
                if hl >= self.min_child_h and hr >= self.min_child_h:
                    gain = 0.5 * (_leaf_score(gl, hl, lam)
                                  + _leaf_score(gr, hr, lam) - parent) - self.gamma
                    if gain > best_gain:
                        best_gain = gain
                        right_idx = np.concatenate([pid[s:], mid])
                        best = (feat, thr, False, pid[:s], right_idx)
                # Pass B -- missing -> left (right = present suffix)
                gr2, hr2 = Gp - csg[s - 1], Hp - csh[s - 1]
                gl2, hl2 = G - gr2, H - hr2
                if hl2 >= self.min_child_h and hr2 >= self.min_child_h:
                    gain = 0.5 * (_leaf_score(gl2, hl2, lam)
                                  + _leaf_score(gr2, hr2, lam) - parent) - self.gamma
                    if gain > best_gain:
                        best_gain = gain
                        left_idx = np.concatenate([pid[:s], mid])
                        best = (feat, thr, True, left_idx, pid[s:])
        return best

    def _predict_one(self, x):
        node = self.root
        while node.feat != -1:
            v = x[node.feat]
            go_left = node.default_left if np.isnan(v) else (v < node.thresh)
            node = node.left if go_left else node.right
        return node.weight

    def predict(self, X):
        return np.array([self._predict_one(X[i]) for i in range(X.shape[0])])


class BoostedTrees:
    """Forward-stagewise second-order boosting; the loss enters only through (g, h)."""

    def __init__(self, n_rounds=100, learning_rate=0.1, max_depth=6,
                 lam=1.0, gamma=0.0, loss="squarederror"):
        self.n_rounds, self.lr, self.max_depth = n_rounds, learning_rate, max_depth
        self.lam, self.gamma, self.loss, self.trees = lam, gamma, loss, []

    def _grad_hess(self, y, yp):
        if self.loss == "squarederror":              # l = 0.5 (y - yp)^2
            return (yp - y), np.ones_like(y)          # g = yp - y, h = 1
        if self.loss == "logistic":                  # binary logloss, y in {0,1}
            p = 1.0 / (1.0 + np.exp(-yp))
            return (p - y), np.maximum(p * (1 - p), 1e-6)   # g = p - y, h = p(1-p)
        raise ValueError(self.loss)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if self.loss == "squarederror":
            self.base = float(np.mean(y))
        else:
            p0 = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
            self.base = float(np.log(p0 / (1 - p0)))
        yp = np.full(len(y), self.base)
        for _ in range(self.n_rounds):
            g, h = self._grad_hess(y, yp)
            tree = RegularizedTree(self.max_depth, self.lam, self.gamma).fit(X, g, h)
            self.trees.append(tree)
            yp += self.lr * tree.predict(X)           # shrinkage
        return self

    def predict(self, X):
        out = np.full(X.shape[0], self.base)
        for tree in self.trees:
            out += self.lr * tree.predict(X)
        return out
```

The production system is the same algorithm with the scalable engine — pre-sorted column blocks, the histogram-binned weighted-quantile `hist` split finder, cache-aware prefetch, out-of-core blocks, and parallel allreduce — underneath, exposed as a scikit-learn-style regressor:

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    objective="reg:squarederror",  # l = 0.5 (y - yhat)^2  =>  g = yhat - y, h = 1
    n_estimators=120,              # boosting rounds (trades off with learning_rate)
    max_depth=3,                   # shallow weak learners
    learning_rate=0.05,            # shrinkage eta
    subsample=0.9,                 # row subsample (stochastic gradient boosting)
    colsample_bytree=0.8,          # column subsample (RandomForest-style)
    reg_lambda=1.0,                # the +lambda in w* = -G/(H+lambda)
    tree_method="hist",            # histogram-binned weighted-quantile split finder
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
