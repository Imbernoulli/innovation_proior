**Problem (from rung 2).** The symbolic forms rescued vocab ($0.929$) and dataconstrained ($0.929$) but
lrbsz stayed at $R^2 = -3.05$ — its `MAE` fell to $0.063$ (an order of magnitude better than kernel
ridge), so the magnitudes are nearly right but the *ranking* is wrong: a single fitted basin center
$(\log l^\star, \log b^\star)$ cannot follow an optimum that drifts with scale. The hand-shaped form has
plateaued on the hard family.

**Key idea.** Go back to a flexible learner, but one whose bias does not collapse off the hull. A boosted
ensemble of regression trees: each split is conditional on the splits above it, so it learns a *different*
effective $(l, b)$ optimum per scale region by branching on $N$/$D$ — the scale-dependent drift the fixed
basin lacked — and off the training hull it *flattens to the boundary value* rather than decaying to zero
like an RBF kernel, a crude version of the floor.

**Why.** Trees represent basins and cross-axis interactions natively (a staircase low-loss region, splits
on $\log l$ inside a branch already split on $N$ = "best lr depends on scale"), so flexibility beats a
rigid basin on lrbsz. The cost is discretization: on the smooth saturating tail of dataconstrained the
staircase extrapolates worse than the explicit effective-token law, so this rung *trades* — win lrbsz,
hold/raise vocab, give back some dataconstrained.

**Scaffold edit / hyperparameters.** Same mixed feature map (raw + log + one-hot group) as rung 1.
XGBoost: `n_estimators=120`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.9`,
`colsample_bytree=0.8`, `reg_lambda=1.0`, `tree_method="hist"`, seeded; `GradientBoostingRegressor`
fallback. Fit in **log-target** space only when $y$ is strictly positive (lrbsz, dataconstrained); fit
the raw target for vocab (the unigram loss can be negative). `num_features_` reported back to the loop.

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
