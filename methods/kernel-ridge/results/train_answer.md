We are given training pairs $(x_1, y_1), \dots, (x_T, y_T)$ with numeric descriptors as inputs and real-valued targets, and we want a supervised regressor that predicts a new target under squared error. Three constraints have to hold at once, and that combination is what defeats the obvious tools. The response surface may be nonlinear, so a straight line in the original coordinates is too rigid when the target curves smoothly with the descriptors. The standard cure for rigidity — build a richer feature vector $\phi(x)$ and fit a linear model there — explodes the dimension: polynomial features grow combinatorially, spline and radial-basis expansions grow larger still, and a radial-basis feature map is naturally infinite-dimensional, so any algorithm that forms or inverts a matrix sized by the feature dimension becomes unusable. And the fit must be stable: with many correlated descriptors, or more features than examples, ordinary least squares becomes singular or high-variance and chases noise.

Walking the baselines makes the gap concrete. Ordinary least squares minimizes $\sum_t (y_t - w^\top x_t)^2$ with solution $w = (X^\top X)^{-1} X^\top y$; it is cheap when $X^\top X$ is well-conditioned but rigid and fragile, since in the SVD $X = U D V^\top$ the small singular values of $D$ inflate coefficient variance and a zero singular value kills the inverse outright. Ridge regression fixes the stability half by minimizing $\alpha \|w\|^2 + \sum_t (y_t - w^\top x_t)^2$, giving $(X^\top X + \alpha I)\,w = X^\top y$; because $X^\top X$ is positive semidefinite and $\alpha I$ is positive definite for $\alpha > 0$, the shifted system is always invertible, and in singular-value coordinates each unstable direction is shrunk by $d_j^2/(d_j^2 + \alpha)$. But ridge by itself is still linear in the supplied coordinates, and running it on explicit nonlinear features reintroduces a solve sized by the feature dimension — exactly the quantity that becomes large or infinite. Kernelized support-vector regression shows kernels can avoid explicit coordinates, but its $\epsilon$-insensitive loss buys sparsity at the price of a quadratic program rather than a one-shot linear solve. Kriging and the Gaussian-process posterior mean produce a kernel-weighted predictor, but live in a Bayesian/geostatistical vocabulary whose tie to a regularized squared-loss fit is left implicit. So the open problem is: keep ridge's stability, let the fitted function be nonlinear in the original input, and make the computation depend on the number of examples, not the feature dimension.

I propose kernel ridge regression — ridge regression carried out in a kernel-induced feature space. The direct nonlinear move is to replace $x$ by $\phi(x)$ and run the ridge calculation there, $w = (\Phi^\top \Phi + \alpha I)^{-1} \Phi^\top y$ with prediction $w^\top \phi(x)$. The matrix being inverted is feature-dimension by feature-dimension, which is the wall. What rescues it is a purely algebraic identity that moves the inverse from feature dimension to example dimension. For compatible rectangular $P$ and $Q$, $(PQ + \alpha I)\,P = P\,(QP + \alpha I)$, so with both shifted matrices invertible, $(PQ + \alpha I)^{-1} P = P\,(QP + \alpha I)^{-1}$. Setting $P = \Phi^\top$ and $Q = \Phi$ turns the feature-space solution into

$$w = \Phi^\top (\Phi \Phi^\top + \alpha I)^{-1} y.$$

Now $\Phi \Phi^\top$ is a $T \times T$ matrix whose $(s,t)$ entry is exactly $\phi(x_s)^\top \phi(x_t)$. Defining $c = (\Phi \Phi^\top + \alpha I)^{-1} y$ gives $w = \sum_t c_t\, \phi(x_t)$: the learned weight vector lies in the span of the training feature vectors, which is also geometrically forced, since any component of $w$ orthogonal to every $\phi(x_t)$ changes no training prediction yet adds to $\|w\|^2$ and is therefore removed by the penalty. A new prediction is $w^\top \phi(x) = \sum_t c_t\, \phi(x_t)^\top \phi(x)$, so the feature map enters only through inner products between examples. That is the opening for the kernel: replace each $\phi(u)^\top \phi(v)$ by a valid positive-semidefinite kernel value $K(u,v)$. Writing the training Gram matrix $K_{s,t} = K(x_s, x_t)$ and the test vector $k(x)_t = K(x_t, x)$, the coefficients satisfy

$$(K + \alpha I)\, c = y, \qquad f(x) = k(x)^\top c = \sum_t c_t\, K(x_t, x).$$

The feature dimension has disappeared from the computation entirely.

It is easy to lose a factor of two here, so the constants are worth pinning down against the constrained form. Minimizing $\alpha \|w\|^2 + \sum_t \xi_t^2$ subject to $y_t - w^\top \phi(x_t) = \xi_t$ with multipliers $\alpha_t$ gives the Lagrangian $\alpha \|w\|^2 + \sum_t \xi_t^2 + \sum_t \alpha_t (y_t - w^\top \phi(x_t) - \xi_t)$. Differentiating in $w$ gives $2\alpha\, w - \sum_t \alpha_t \phi(x_t) = 0$, so $w = \tfrac{1}{2\alpha} \sum_t \alpha_t \phi(x_t)$; differentiating in $\xi_t$ gives $\xi_t = \alpha_t/2$. Substituting back leaves a dual whose stationarity condition is $(K + \alpha I)\,\alpha = 2\alpha\, y$, i.e. the multipliers are $\alpha = 2\alpha (K + \alpha I)^{-1} y$. These multipliers are not what I store: when I form the prediction the prefactor from $w$ cancels the extra $2\alpha$, since $w^\top \phi(x) = \tfrac{1}{2\alpha} k(x)^\top \alpha = k(x)^\top (K + \alpha I)^{-1} y$. So the stored coefficient vector is $c = \alpha/(2\alpha) = (K + \alpha I)^{-1} y$, matching the push-through derivation in both sign and constants.

The function-space view explains what happens when the Gram matrix is singular. In the reproducing-kernel Hilbert space generated by $K$, minimize $\sum_t (y_t - f(x_t))^2 + \alpha \|f\|^2$. Any candidate splits into a part in $\mathrm{span}\{K(\cdot, x_t)\}$ and a part orthogonal to it; the reproducing property says the orthogonal part changes no training value while Pythagoras says it only adds to the norm, so a minimizer is $f = \sum_t c_t K(\cdot, x_t)$. For that representation the training values are $Kc$ and the norm is $c^\top K c$, so the objective is $\|y - Kc\|^2 + \alpha\, c^\top K c$ with gradient $2K\big((K + \alpha I)c - y\big)$. If $K$ is invertible this gives $(K + \alpha I)c = y$ at once. If $K$ is singular I must not cancel $K$: diagonalizing $K = U \,\mathrm{diag}(d_i)\, U^\top$ and writing $c = Uz$, $y = Ur$, each coordinate contributes $(r_i - d_i z_i)^2 + \alpha d_i z_i^2$, so for $d_i > 0$ the minimizer obeys $(d_i + \alpha) z_i = r_i$, and for $d_i = 0$ the coordinate touches neither the fitted values nor the norm and represents the zero function. The shifted system $(K + \alpha I)c = y$ simply selects one valid representative; others differ only in nullspace directions and predict identically.

Several design choices earn their place. The squared loss is what makes this a single linear solve: the derivative of the squared residual is linear in the fitted values, so the representer step yields one linear system, whereas the $\epsilon$-insensitive support-vector loss would buy sparsity at the cost of a quadratic program — for a medium-sized regression, a dense coefficient vector and one stable solve is the simpler trade. The diagonal shift $\alpha I$ is not cosmetic: a Gram matrix is positive semidefinite but goes singular when examples coincide in feature space and ill-conditioned when they are close, so with $\alpha = 0$ the solve $Kc = y$ demands exact interpolation and amplifies noise in small-eigenvalue directions, while adding $\alpha I$ lifts every eigenvalue by $\alpha$ and shrinks the fit toward zero. This same $\alpha$ has a probabilistic reading: with a zero-mean Gaussian prior of covariance $\tfrac{1}{2\alpha} I$ on $w$ and observation-noise variance $\tfrac{1}{2}$, Gaussian conditioning reproduces the posterior mean $k(x)^\top (K + \alpha I)^{-1} y$, so $\alpha$ is the noise-to-prior-variance ratio. For the nonlinear similarity itself, the radial-basis kernel $K(x,y) = \exp(-\gamma \|x - y\|^2)$ is the clean default: symmetric and positive semidefinite for $\gamma > 0$, corresponding to an infinite-dimensional expansion, with $\gamma$ as a single bandwidth knob — large $\gamma$ gives narrow, local bumps and small $\gamma$ gives broad, smooth ones. Because Euclidean distance is scale-sensitive, the numeric inputs have to be put on comparable scales first; a log-transformed copy exposes multiplicative structure while standardized raw features keep ordinary linear trends, and categorical descriptors must become numeric (one-hot) before they can enter the distance. The implementation therefore separates preprocessing from the core algorithm: the feature map is fitted only on training inputs, the RBF bandwidth is set once the final feature dimension is known, and the canonical estimator does the real work — form $K$, add its regularization constant to the diagonal, solve for $\mathrm{dual\_coef\_} = (K + \alpha I)^{-1} y$, store the training rows, and predict by the cross-kernel matrix times $\mathrm{dual\_coef\_}$.

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
