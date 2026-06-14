# XGBoost, distilled

XGBoost is a scalable end-to-end gradient tree boosting system built on a **regularized,
second-order objective**. Each boosting round adds one regression tree that minimizes a
second-order Taylor approximation of the loss plus an explicit complexity penalty; the loss
enters only through its first and second derivatives `(g_i, h_i)`, so any differentiable convex
objective plugs into the same engine. The accuracy comes from the regularized Newton-style
objective; the scalability comes from a sparsity-aware split finder, a provably-bounded weighted
quantile sketch for approximate splits, pre-sorted column blocks, cache-aware access, and
out-of-core/distributed execution.

## Problem it solves

Tree boosting is the consensus high-accuracy learner on structured/tabular data, but existing
systems do not scale: they re-sort data each iteration, are in-memory only, handle sparsity with
ad-hoc hacks, and lack a rigorous weighted-quantile primitive for approximate split finding —
while also keeping regularization outside the split criterion. XGBoost is one system that is both
as accurate as the best boosting and able to push billions of sparse rows through limited
resources.

## Key idea

Model: a tree ensemble `ŷ_i = Σ_{k=1}^K f_k(x_i)`, each `f_k(x) = w_{q(x)}` (`q` routes `x` to
one of `T` leaves; `w ∈ ℝ^T` are continuous leaf scores). Optimize the regularized objective

```
L(φ) = Σ_i l(ŷ_i, y_i) + Σ_k Ω(f_k),    Ω(f) = γ T + ½ λ ‖w‖².
```

`γ` charges per leaf (complexity + pruning threshold), `λ` is L2 on leaf scores. Fit additively:
at round `t`, add `f_t` minimizing `Σ_i l(y_i, ŷ_i^{(t-1)} + f_t(x_i)) + Ω(f_t)`. Second-order
Taylor about `ŷ^{(t-1)}` and dropping the constant loss term gives

```
L̃^(t) = Σ_i [ g_i f_t(x_i) + ½ h_i f_t²(x_i) ] + Ω(f_t),
g_i = ∂_{ŷ} l(y_i, ŷ^{(t-1)}),   h_i = ∂²_{ŷ} l(y_i, ŷ^{(t-1)}).
```

The objective sees the loss only through `(g_i, h_i)` → arbitrary losses (regression, logistic,
ranking, custom) share one engine.

## Optimal leaf weight, structure score, split gain

Group instances by leaf, `I_j = {i : q(x_i) = j}`, with `G_j = Σ_{i∈I_j} g_i`,
`H_j = Σ_{i∈I_j} h_i`. Per leaf the objective is the convex quadratic
`G_j w_j + ½ (H_j + λ) w_j²`, minimized at

```
w_j* = - G_j / (H_j + λ),          (Newton step, with λ damping low-curvature leaves)
```

and substituting back gives the **structure score** of a fixed tree `q`:

```
L̃^(t)(q) = - ½ Σ_{j=1}^T G_j² / (H_j + λ) + γ T.
```

Lower is better; this is "impurity for an arbitrary loss, with complexity included." Splitting a
node `I` into `I_L, I_R` (where `G = G_L + G_R`, `H = H_L + H_R`) reduces the loss by

```
Gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ.
```

If the best `Gain <= 0`, don't split: the loss reduction has not paid for the extra leaf. Thus
`γ` is the built-in pre-pruning threshold. **Squared-error special case:** `l = ½(y−ŷ)²` ⇒ `g_i = ŷ_i − y_i`,
`h_i = 1`, so `w_j*` = (regularized) mean residual and the score reduces to the classical
first-order picture — the construction contains gradient boosting.

## Split finding

- **Exact greedy (Alg. 1):** for each feature, sort instances, scan once accumulating `G_L, H_L`
  (`G_R = G − G_L`, `H_R = H − H_L`), track max `Gain`. Considers every split point.
- **Approximate (Alg. 2):** propose candidate split points per feature (global per-tree or local
  per-split), bucket instances, aggregate `(G, H)` per bucket, search bucket boundaries. Needed
  for out-of-core / distributed. Completing the square,
  `L̃^(t) = Σ_i ½ h_i (f_t(x_i) − (−g_i/h_i))² + const`, shows it is weighted squared error with
  weight `h_i`; so candidates should be **`h`-weighted quantiles**:
  `r_k(z) = (Σ_{x<z} h)/(Σ h)`, with `|r_k(s_j) − r_k(s_{j+1})| < ε` (≈ `1/ε` candidates).
  When `h_i ≡ 1` this is ordinary quantiles.
- **Sparsity-aware (Alg. 3):** each node has a **default direction**; missing/absent values flow
  the default way. Learn it from the same gain by two passes over only the non-missing entries —
  ascending with the missing block sent right, descending with it sent left — keep the better.
  Cost is linear in the number of non-zeros.

## Weighted quantile sketch (provable `ε`-guarantee)

For weighted data `D = {(x_i, w_i)}`, with `r⁻(y)=Σ_{x<y}w`, `r⁺(y)=Σ_{x≤y}w`, `ω(y)=r⁺−r⁻`,
a summary `Q=(S, r̃⁺, r̃⁻, ω̃)` is `ε`-approximate iff `r̃⁺(y) − r̃⁻(y) − ω̃(y) ≤ ε ω(D)` ∀`y`
(equivalently, two discrete conditions per consecutive pair). Two operations preserve it:

- **Merge** two summaries (add the extended functions pointwise): error becomes `max(ε_1, ε_2)`
  — does **not** accumulate, because `r̃⁺−r̃⁻−ω̃` adds and `ω(D)=ω(D_1)+ω(D_2)`.
- **Prune** to `b+1` points (query the summary at ranks `(i−1)/b·ω(D)` via a midpoint query
  whose returned point brackets the target rank to `±(ε/2)ω(D)`): error becomes `ε + 1/b`.

Same guarantee shape as the classical *unweighted* Greenwald–Khanna sketch, now for arbitrary
weights — and the weights are exactly the curvatures `h_i`. Merges keep error flat, prunes add
`1/b`, so a streaming/distributed pipeline keeps total error at a controllable `ε`.

## Regularization and system design

- **Shrinkage** `η`: `F_t = F_{t-1} + η f_t`, `0<η<1` (learning rate; trades off with #trees).
- **Column subsampling** (RandomForest-style) and **row subsampling** (stochastic GB):
  decorrelate trees, reduce overfit, speed up.
- **Column block (CSC, pre-sorted once):** removes the per-iteration sort (saves a `log n`
  factor), enables one-pass split finding for all leaves, parallel per-column statistics, and
  cheap column subsampling.
- **Cache-aware prefetch:** the by-row `(g, h)` fetches during a feature-sorted scan are
  non-contiguous; prefetch into a per-thread buffer and accumulate from it. Block size ≈ `2^16`
  balances parallelism vs. cache.
- **Out-of-core:** disk-resident blocks with an independent prefetch thread, per-column
  compression decompressed on the fly, and sharding across disks.

## Final algorithm

```
F_0 = argmin_c Σ_i l(y_i, c)
for t = 1..M:
    g_i = ∂_{ŷ} l(y_i, F_{t-1}(x_i)),   h_i = ∂²_{ŷ} l(y_i, F_{t-1}(x_i))     # for all i
    grow a regression tree on (g, h):
        at each node split by argmax Gain = ½[G_L²/(H_L+λ)+G_R²/(H_R+λ)−G²/(H+λ)] − γ
        (candidates = exact scan, or h-weighted-quantile buckets; missing -> learned default)
        keep a split only if Gain > 0
        leaf weight  w_j* = − G_j / (H_j + λ)
    F_t = F_{t-1} + η · tree_t                                                # shrinkage
```

## Working code

A faithful from-scratch core of the regularized second-order tree (the structure score, the
split gain, the optimal leaf weight, and the sparsity-aware learned default direction), plus the
forward-stagewise boosting loop in which the loss enters only through `(g, h)`:

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

The production system is the same algorithm with the scalable engine (pre-sorted column blocks,
the histogram-binned weighted-quantile `hist` split finder, cache-aware prefetch, out-of-core
blocks, parallel allreduce) underneath, exposed as a scikit-learn-style regressor:

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
