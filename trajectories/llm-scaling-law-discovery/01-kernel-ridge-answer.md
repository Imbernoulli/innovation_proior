**Problem.** The scaffold default predicts the train-target mean and so sits at the trivial floor
($R^2 \le 0$ off-distribution). The first real rung asks the cheapest, most trustworthy question: how
much of each family's loss surface is just smooth structure in the descriptors, recoverable by a
flexible regressor with *no* scaling-law prior baked in?

**Key idea.** Kernel ridge regression: the regularized least-squares fit $w = (\Phi'\Phi + aI)^{-1}\Phi'y$
re-expressed by the push-through identity as $f(x) = \sum_t c_t K(x_t, x)$ with $(K + aI)c = y$, so the
cost depends on the number of runs $T$, not the feature dimension. Use the universal Gaussian RBF kernel
$K(x,y) = \exp(-\gamma\lVert x-y\rVert^2)$ with one bandwidth knob, and feed it a mixed feature map —
standardized raw numerics **plus** standardized $\log(1+\cdot)$ numerics (to capture the power-law
geometry) **plus** a one-hot of the `group` (so the single regressor still separates families).

**Why it is the weakest rung.** An RBF kernel is pure locality: $K(x, x_t)\to 0$ as the query leaves the
training hull, so on the larger/denser held-out region the prediction collapses toward zero with no
power-law tail and no irreducible floor. It learns the interior and has nothing to say about the
boundary — which is exactly what the extrapolation test scores. Its predicted off-hull failure is what
forces the next rungs to impose the right asymptotic *form* rather than more flexibility.

**Scaffold edit / hyperparameters.** RBF kernel ridge; $a = 0.05$ (alpha), $\gamma = 1/n_{\text{features}}$;
target fit **raw** (no log on $y$ — the vocab target can be negative); `include_raw=True`,
`include_log=True`; one-hot categoricals. `num_features_` is reported back to the loop.

```python
from sklearn.kernel_ridge import KernelRidge as _KernelRidge


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
    """Black-box kernel ridge baseline on mixed SLDBench features."""

    def __init__(self, benchmark_name, numeric_names=None,
                 categorical_names=None):
        self.benchmark_name = benchmark_name
        self.encoder = _FeatureMap(include_raw=True, include_log=True)
        self.model = None
        self.num_features_ = 0

    def fit(self, X_num, X_cat, y):
        features = self.encoder.fit_transform(X_num, X_cat)
        gamma = 1.0 / max(features.shape[1], 1)
        self.model = _KernelRidge(alpha=0.05, kernel="rbf", gamma=gamma)
        self.model.fit(features, np.asarray(y, dtype=float))
        self.num_features_ = features.shape[1]
        return self

    def predict(self, X_num, X_cat):
        features = self.encoder.transform(X_num, X_cat)
        return np.asarray(self.model.predict(features), dtype=float)
```
