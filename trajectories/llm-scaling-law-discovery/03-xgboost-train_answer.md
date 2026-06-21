The symbolic rung did what I predicted on the easy families and what I feared on the hard one. Vocab swung from $-0.567$ to $0.929$ and dataconstrained from $-13.4$ to $0.929$ — both rescued, because the power-law-plus-floor backbone and the effective-token term finally gave the law something that extrapolates to the larger/denser test points. But lrbsz came back at $R^2=-3.05$, still worse than predicting the mean, even though its `MAE` dropped from $0.619$ to $0.063$, an order of magnitude. So the hand-shaped basin is *close* in magnitude but gets the *ranking* of held-out points wrong: a single fitted center $(\log l^\star,\log b^\star)$ cannot follow an optimum that drifts with scale, so out-of-sample the bowl is centered in the wrong place and the quadratic penalty has the wrong curvature there. Tweaking a hand-shaped form will not fix this, because the problem is the fixed center itself.

That reopens the question I closed at the first rung. Kernel ridge collapsed off the hull for a *specific* reason — RBF locality, kernel values decaying to zero far from training points, no floor — not because flexible learners are wrong here. So the right question is whether there is a flexible, model-free learner whose inductive bias does *not* collapse off the hull and that can represent a scale-dependent basin without my hand-shaping a single center. I propose **gradient-boosted regression trees (XGBoost)**.

A regression tree partitions the input space into axis-aligned cells and predicts a constant per cell — crude and high-variance alone. Boosting builds an additive ensemble $\hat y_i=\sum_k f_k(x_i)$ where each tree corrects the residual of the ensemble so far, and the modern formulation regularizes the whole objective: minimize $\sum_i l(\hat y_i,y_i)+\sum_k\Omega(f_k)$ with $\Omega(f)=\gamma T+\tfrac12\lambda\lVert w\rVert^2$ — a per-leaf cost $\gamma T$ and an $L_2$ penalty on leaf scores. Fitting is additive: at round $t$, second-order Taylor-expand the loss around the current prediction so the per-round objective depends on the loss only through the gradient $g_i$ and curvature $h_i$. The optimal leaf weight is then

$$w_j^\star=-\frac{G_j}{H_j+\lambda},$$

with $G_j,H_j$ the gradient and curvature sums in the leaf — a Newton step damped by $\lambda$ so an under-populated leaf cannot blow up. Every candidate split is scored by the gain $\tfrac12\!\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{G^2}{H+\lambda}\right]-\gamma$, where $\gamma$ doubles as the prune threshold: a split that does not clear $\gamma$ is not made.

Why this fits where kernel ridge did not comes down to two structural facts. First, the off-hull bias is fundamentally different. A tree does not decay to zero away from training points — it predicts the constant of whichever leaf the query lands in, which for an out-of-region query is the constant of the *nearest boundary cell*. So instead of collapsing toward zero, a boosted-tree prediction *flattens to the last seen value* in that direction; on a loss surface that saturates toward a floor, "flatten to the boundary value" is a far better extrapolation than "decay to zero" — a crude version of the floor the symbolic law imposes explicitly. Second — and this is what could crack lrbsz — trees represent basins and cross-axis interactions *natively*, with no hand-shaped center. The basin in $(\log l,\log b)$ is just a region of low loss surrounded by higher loss; an axis-aligned partition carves it into cells and assigns each its own constant, reconstructing the bowl as a staircase without ever fitting a quadratic or its center. And because every split is conditional on the splits above it, interactions come for free: a split on $\log l$ *inside* a branch already split on $N$ is precisely "the best learning rate depends on the model scale" — the scale-dependent optimum drift that broke the symbolic law. The tree learns a different effective optimum in each scale region by branching, with no closed-form $l^\star(N,D)$.

The same staircase that helps off the hull also *costs* me where the symbolic law already won. A power-law decay is smooth and monotone; a tree approximates it with a finite staircase, so on a family that is a clean additive power law the tree pays a discretization error the exact symbolic form does not. So this rung is a *trade*: I expect to win lrbsz (flexibility beats a rigid hand-shaped basin), hold or raise vocab (smooth surface near the hull, where the discretization error is small), and *give back* some dataconstrained — past the training hull, the test points are denser and the staircase just holds the boundary constant where the explicit saturating effective-token law keeps bending toward the floor.

Concretely, in the task's edit surface I reuse the same mixed feature map as the kernel-ridge rung — standardized raw numerics plus standardized $\log(1+\cdot)$ numerics plus a one-hot of `group` — because the log features give the trees the power-law geometry to split on and the one-hot lets the single ensemble separate families by branching on the group indicator. I fit shallow trees (`max_depth=3`, weak learners), many rounds (`n_estimators=120`) at a small learning rate (`0.05`, Friedman shrinkage so no single tree dominates), with row subsampling (`subsample=0.9`, stochastic boosting) and column subsampling (`colsample_bytree=0.8`, decorrelating the trees) and the $L_2$ leaf penalty (`reg_lambda=1.0`, the $+\lambda$ in $w^\star$); the `hist` tree method uses the weighted-quantile-binned split finder. One detail is load-bearing and is where this rung diverges from a textbook regressor: I fit in *log-target* space **only when the target is strictly positive** — for lrbsz's lm_loss and dataconstrained's loss, fitting $\log y$ gives the trees a multiplicative error scale that matches a power-law surface — but the vocab target is a unigram-normalised loss that *can be negative*, so for vocab I fall back to fitting $y$ directly in the linear domain. That conditional is the entire signed-target handling, mirroring the linear-vs-log residual choice the symbolic rung made per family. I also keep a `GradientBoostingRegressor` fallback with matching hyperparameters so the rung runs even when the boosted-tree package is unavailable.

The falsifiable claims against the symbolic numbers: lrbsz should improve in $R^2$ from $-3.05$ toward $-1$ (still likely negative — the held-out region is genuinely hard and the staircase cannot fully extrapolate, but conditional splits should cut the ranking error the fixed basin made) with `MAE` dropping further below $0.063$; vocab should *rise* above $0.929$ toward the high $0.97$s; and dataconstrained should *fall* below $0.929$ into the mid-$0.8$s. If that is the pattern — lrbsz and vocab won, dataconstrained given back — then the lesson is sharp: neither the rigid hand-shaped form nor the asymptotically-blind tree dominates, and the strongest solution is the one that carries the *correct literature-grounded asymptotic form per family*, including a scale-dependent lrbsz optimum that this tree learns only by staircase and that the symbolic basin lacked entirely.

```python
try:
    from xgboost import XGBRegressor as _XGBRegressor
except Exception:
    _XGBRegressor = None
from sklearn.ensemble import GradientBoostingRegressor as _GBR


class _FeatureMap:
    """Mixed numeric/categorical encoder for black-box baselines."""

    def __init__(self, include_raw=True, include_log=True):
        self.include_raw = include_raw
        self.include_log = include_log

    def fit(self, X_num, X_cat):
        X_num = np.asarray(X_num, dtype=float)
        self.num_medians_ = np.nanmedian(X_num, axis=0)
        self.num_medians_ = np.where(np.isnan(self.num_medians_), 0.0,
                                     self.num_medians_)
        filled = np.where(np.isnan(X_num), self.num_medians_, X_num)
        self.raw_mean_ = filled.mean(axis=0)
        self.raw_std_ = filled.std(axis=0)
        self.raw_std_[self.raw_std_ < 1e-8] = 1.0
        clipped = np.clip(filled, a_min=0.0, a_max=None)
        logged = np.log1p(clipped)
        self.log_mean_ = logged.mean(axis=0)
        self.log_std_ = logged.std(axis=0)
        self.log_std_[self.log_std_ < 1e-8] = 1.0
        self.cat_levels_ = []
        X_cat = np.asarray(X_cat, dtype=object)
        for col in range(X_cat.shape[1]):
            values = [str(v) if v is not None else "__MISSING__"
                      for v in X_cat[:, col]]
            self.cat_levels_.append(sorted(set(values)))
        return self

    def _transform_num(self, X_num):
        X_num = np.asarray(X_num, dtype=float)
        filled = np.where(np.isnan(X_num), self.num_medians_, X_num)
        pieces = []
        if self.include_raw:
            pieces.append((filled - self.raw_mean_) / self.raw_std_)
        if self.include_log:
            logged = np.log1p(np.clip(filled, a_min=0.0, a_max=None))
            pieces.append((logged - self.log_mean_) / self.log_std_)
        return np.concatenate(pieces, axis=1) if pieces else filled

    def _transform_cat(self, X_cat):
        X_cat = np.asarray(X_cat, dtype=object)
        if X_cat.shape[1] == 0:
            return np.empty((X_cat.shape[0], 0), dtype=float)
        cols = []
        for col, levels in enumerate(self.cat_levels_):
            values = [str(v) if v is not None else "__MISSING__"
                      for v in X_cat[:, col]]
            onehot = np.zeros((X_cat.shape[0], len(levels)), dtype=float)
            level_to_idx = {level: idx for idx, level in enumerate(levels)}
            for row_idx, value in enumerate(values):
                idx = level_to_idx.get(value)
                if idx is not None:
                    onehot[row_idx, idx] = 1.0
            cols.append(onehot)
        return np.concatenate(cols, axis=1)

    def transform(self, X_num, X_cat):
        num = self._transform_num(X_num)
        cat = self._transform_cat(X_cat)
        if cat.size == 0:
            return num
        if num.size == 0:
            return cat
        return np.concatenate([num, cat], axis=1)

    def fit_transform(self, X_num, X_cat):
        return self.fit(X_num, X_cat).transform(X_num, X_cat)


class ScalingLawModel:
    """Boosted-tree baseline on mixed SLDBench features."""

    def __init__(self, benchmark_name, numeric_names=None,
                 categorical_names=None):
        self.benchmark_name = benchmark_name
        self.encoder = _FeatureMap(include_raw=True, include_log=True)
        self.model = None
        self.num_features_ = 0

    def fit(self, X_num, X_cat, y):
        seed = int(os.environ.get("SEED", "42"))
        features = self.encoder.fit_transform(X_num, X_cat)
        y_arr = np.asarray(y, dtype=float)
        # Fit in log-space only when the target is strictly positive (e.g.
        # lm_loss, training loss). Targets like vocab unigram-normalised loss
        # can be negative, so fall back to linear regression in that case.
        self._use_log = bool(np.all(y_arr > 0.0))
        target = np.log(np.clip(y_arr, EPS, None)) if self._use_log else y_arr
        if _XGBRegressor is not None:
            self.model = _XGBRegressor(
                objective="reg:squarederror",
                n_estimators=120, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                tree_method="hist", n_jobs=4, verbosity=0, random_state=seed,
            )
        else:
            self.model = _GBR(
                n_estimators=120, learning_rate=0.05, max_depth=3,
                random_state=seed,
            )
        self.model.fit(features, target)
        self.num_features_ = features.shape[1]
        return self

    def predict(self, X_num, X_cat):
        features = self.encoder.transform(X_num, X_cat)
        raw = np.asarray(self.model.predict(features), dtype=float)
        return np.exp(raw) if getattr(self, "_use_log", False) else raw
```
