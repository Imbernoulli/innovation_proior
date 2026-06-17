# Kernel Ridge Regression, Distilled

Kernel ridge regression is ridge regression in a kernel-induced feature space. With training Gram matrix `K_{s,t} = K(x_s, x_t)` and regularization `a > 0`, it stores

```text
c = (K + a I)^-1 y
```

and predicts

```text
f(x) = sum_t c_t K(x_t, x) = k(x)^T c.
```

The Lagrange multipliers in the constrained derivation are `alpha = 2a(K+aI)^-1 y`; implementation coefficients are `c = alpha/(2a)`, so no `2a` appears in prediction.

## Why This Is The Method

- Ridge's `aI` stabilizes least squares by lifting the spectrum and shrinking small-eigenvalue directions.
- The identity `(Phi^T Phi + aI)^-1 Phi^T = Phi^T(Phi Phi^T + aI)^-1` moves the solve from feature dimension to sample dimension.
- The representer theorem gives the same finite form: the minimizer of `sum_t(y_t-f(x_t))^2 + a||f||^2` lies in `span{K(., x_t)}`. If `K` is singular, coefficients need not be unique, but nullspace changes represent the zero function.
- Squared loss gives a single linear system. Epsilon-insensitive support-vector regression uses the same expansion form but solves a quadratic program and gets sparsity instead of a closed form.
- With a Gaussian prior and Gaussian noise, the same formula is the Kriging/Gaussian-process posterior mean, with `a` acting as a noise-to-prior-variance ratio.

## Scaffold Code

This version keeps the core solve faithful to the canonical scikit-learn implementation by using `sklearn.kernel_ridge.KernelRidge` directly. The feature map is task-specific preprocessing around that estimator.

```python
import numpy as np
from sklearn.kernel_ridge import KernelRidge as _KernelRidge


class _FeatureMap:
    """Mixed numeric/categorical encoder for an RBF kernel baseline."""

    def __init__(self, include_raw=True, include_log=True):
        self.include_raw = include_raw
        self.include_log = include_log

    @staticmethod
    def _infer_rows(X_num, X_cat):
        for X in (X_num, X_cat):
            if X is None:
                continue
            X = np.asarray(X)
            if X.size == 0:
                continue
            if X.ndim == 0:
                return 1
            return X.shape[0]
        return 0

    @staticmethod
    def _num2d(X, n_rows=None):
        if X is None:
            return np.empty((0 if n_rows is None else n_rows, 0), dtype=float)

        X = np.asarray(X, dtype=float)
        if X.size == 0:
            return np.empty((0 if n_rows is None else n_rows, 0), dtype=float)
        if X.ndim == 0:
            X = X.reshape(1, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if n_rows is not None and X.shape[0] != n_rows:
            raise ValueError("Numeric and categorical inputs must have the same row count.")
        return X

    @staticmethod
    def _cat2d(X, n_rows):
        if X is None:
            return np.empty((n_rows, 0), dtype=object)

        X = np.asarray(X, dtype=object)
        if X.size == 0:
            return np.empty((n_rows, 0), dtype=object)
        if X.ndim == 0:
            X = X.reshape(1, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != n_rows:
            raise ValueError("Numeric and categorical inputs must have the same row count.")
        return X

    def fit(self, X_num, X_cat):
        n_rows = self._infer_rows(X_num, X_cat)
        X_num = self._num2d(X_num, n_rows)
        X_cat = self._cat2d(X_cat, n_rows)

        if X_num.shape[1]:
            self.num_medians_ = np.nanmedian(X_num, axis=0)
            self.num_medians_ = np.where(np.isnan(self.num_medians_), 0.0, self.num_medians_)
            filled = np.where(np.isnan(X_num), self.num_medians_, X_num)

            self.raw_mean_ = filled.mean(axis=0)
            self.raw_std_ = filled.std(axis=0)
            self.raw_std_[self.raw_std_ < 1e-8] = 1.0

            logged = np.log1p(np.clip(filled, a_min=0.0, a_max=None))
            self.log_mean_ = logged.mean(axis=0)
            self.log_std_ = logged.std(axis=0)
            self.log_std_[self.log_std_ < 1e-8] = 1.0
        else:
            self.num_medians_ = np.empty(0)
            self.raw_mean_ = self.raw_std_ = np.empty(0)
            self.log_mean_ = self.log_std_ = np.empty(0)

        self.cat_levels_ = []
        for col in range(X_cat.shape[1]):
            values = [str(v) if v is not None else "__MISSING__" for v in X_cat[:, col]]
            self.cat_levels_.append(sorted(set(values)))
        return self

    def _transform_num(self, X_num, n_rows):
        X_num = self._num2d(X_num, n_rows)
        if X_num.shape[1] != self.num_medians_.shape[0]:
            raise ValueError("Numeric input width differs from the fitted data.")
        if X_num.shape[1] == 0:
            return np.empty((X_num.shape[0], 0), dtype=float)

        filled = np.where(np.isnan(X_num), self.num_medians_, X_num)
        pieces = []
        if self.include_raw:
            pieces.append((filled - self.raw_mean_) / self.raw_std_)
        if self.include_log:
            logged = np.log1p(np.clip(filled, a_min=0.0, a_max=None))
            pieces.append((logged - self.log_mean_) / self.log_std_)
        return np.concatenate(pieces, axis=1) if pieces else np.empty((X_num.shape[0], 0))

    def _transform_cat(self, X_cat, n_rows):
        X_cat = self._cat2d(X_cat, n_rows)
        if X_cat.shape[1] != len(self.cat_levels_):
            raise ValueError("Categorical input width differs from the fitted data.")
        cols = []
        for col, levels in enumerate(self.cat_levels_):
            values = [str(v) if v is not None else "__MISSING__" for v in X_cat[:, col]]
            onehot = np.zeros((X_cat.shape[0], len(levels)), dtype=float)
            level_to_idx = {level: idx for idx, level in enumerate(levels)}
            for row_idx, value in enumerate(values):
                idx = level_to_idx.get(value)
                if idx is not None:
                    onehot[row_idx, idx] = 1.0
            cols.append(onehot)
        return np.concatenate(cols, axis=1) if cols else np.empty((n_rows, 0), dtype=float)

    def transform(self, X_num, X_cat):
        n_rows = self._infer_rows(X_num, X_cat)
        num = self._transform_num(X_num, n_rows)
        cat = self._transform_cat(X_cat, n_rows)
        return np.concatenate([num, cat], axis=1)

    def fit_transform(self, X_num, X_cat):
        return self.fit(X_num, X_cat).transform(X_num, X_cat)


class ScalingLawModel:
    """Black-box RBF kernel ridge baseline on mixed scaling-law descriptors."""

    def __init__(self, benchmark_name, numeric_names=None, categorical_names=None):
        self.benchmark_name = benchmark_name
        self.numeric_names = numeric_names or []
        self.categorical_names = categorical_names or []
        self.encoder = _FeatureMap(include_raw=True, include_log=True)
        self.model = None
        self.num_features_ = 0

    def fit(self, X_num, X_cat, y):
        features = self.encoder.fit_transform(X_num, X_cat)
        self.num_features_ = features.shape[1]
        if self.num_features_ == 0:
            raise ValueError("At least one numeric or categorical feature is required.")
        gamma = 1.0 / max(self.num_features_, 1)
        self.model = _KernelRidge(alpha=0.05, kernel="rbf", gamma=gamma)
        self.model.fit(features, np.asarray(y, dtype=float))
        return self

    def predict(self, X_num, X_cat):
        if self.model is None:
            raise RuntimeError("Call fit before predict.")
        features = self.encoder.transform(X_num, X_cat)
        return np.asarray(self.model.predict(features), dtype=float)
```

The canonical estimator then performs the actual method: build `K`, add its `alpha` regularization parameter to the diagonal, solve for `dual_coef_`, store `X_fit_`, and predict with the cross-kernel matrix times `dual_coef_`.
