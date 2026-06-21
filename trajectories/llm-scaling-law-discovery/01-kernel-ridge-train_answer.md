The scaffold I start from predicts the train-target mean for every test point. By construction that ignores every descriptor and every group, so on the held-out split — which is deliberately the *larger and denser* configurations, an extrapolation region — it lands at the trivial-mean floor: $R^2$ at or below zero, with `MAE`/`RMSE` equal to the test target's own spread. The honest first question is therefore not "what is the cleverest scaling law" but "how much of each family's loss surface is just smooth structure in the descriptors, recoverable without any scaling-law prior baked in." That tells me how much of the variance is plain input geometry versus how much genuinely requires the right asymptotic form — and the cleanest way to answer it is the cheapest *trustworthy* flexible regressor I can write down, one that bends to the data without my imposing any belief about floors or power-law tails.

I propose **kernel ridge regression** on the engineered features. Build it from the bottom. A plain least-squares line $w=(X'X)^{-1}X'y$ fails twice here: the loss surfaces are *curved* (loss versus $\log N$ or $\log D$ is a decaying power, not a line), and the fit is unstable because the descriptors are collinear in these grids — `params` and `tokens` move together, a singular value of $X$ collapses, and $X'X$ is singular. The stability problem has a clean cure: add a penalty on $\lVert w\rVert^2$ and minimize $a\lVert w\rVert^2 + \sum_t (y_t - w\cdot x_t)^2$, which gives $w=(X'X+aI)^{-1}X'y$. Now $X'X+aI$ is strictly positive definite and always invertible, and in the eigenbasis each coordinate is scaled by $d_j^2/(d_j^2+a)$ — near $1$ for the big directions, near $0$ for the unstable tiny ones, trading a little bias for a large drop in variance. That is ridge, and it fixes stability but not rigidity: it is still linear in $x$.

The route to nonlinearity is to feed richer derived features $\phi(x)$ and run ridge on them, $w=(\Phi'\Phi+aI)^{-1}\Phi'y$. But the feature space I actually want — the Gaussian bump $\exp(-\gamma\lVert x-y\rVert^2)$ — has an *infinite*-dimensional map, so the primal solution is not even a finite object, and even a finite polynomial map of dimension $\sim n^d$ makes forming and inverting $\Phi'\Phi+aI$ hopeless. What rescues it is the push-through identity. For $P=\Phi'$, $Q=\Phi$, the algebra $(PQ+aI)P=P(QP+aI)$ rearranges to

$$(\Phi'\Phi+aI)^{-1}\Phi' = \Phi'(\Phi\Phi'+aI)^{-1},\qquad w=\Phi'(\Phi\Phi'+aI)^{-1}y.$$

The inverse is now $T\times T$ — its $(s,t)$ entry is $\phi(x_s)\cdot\phi(x_t)$, one number per pair of training examples — no matter how huge or infinite the feature space. With only a few hundred runs per group, $T$ is tiny. And because $w=\sum_t\alpha_t\,\phi(x_t)$ is automatically a combination of the training feature vectors (any component orthogonal to all $\phi(x_t)$ adds to $\lVert w\rVert^2$ but nothing to predictions, so the penalty kills it), $\phi$ enters *only* through inner products: the prediction at a new $x$ is $\sum_t c_t\,K(x_t,x)$ with $K(u,v)=\phi(u)\cdot\phi(v)$. So I never need $\phi$ itself, only the kernel $K$. With the kernel matrix $K_{s,t}=K(x_s,x_t)$ the coefficients solve $(K+aI)c=y$, a single regularized linear system, and the prediction is $\sum_t c_t\,K(x_t,x)$.

Squared loss is the load-bearing choice: it is exactly what makes the optimality condition *linear* in the unknowns, so the dual reduces to the one-shot solve $(K+aI)c=y$ — an $\varepsilon$-insensitive or margin loss would turn this into a quadratic program with no closed form and an iterative solver, which I do not want for a few hundred examples. The regularizer $a$ is the interpolation-to-smoothing dial: $a\to 0$ forces $f(x_t)=y_t$ exactly, with coefficients exploding where $K$ is near-singular; lifting every eigenvalue of $K$ by $a$ makes $K+aI$ strictly positive definite (a stable Cholesky exists) and shrinks the fit toward the prior mean — read the Bayesian way, $a$ is the noise-to-prior ratio and the solution is the Gaussian-process posterior mean. For the kernel I take the Gaussian RBF $K(x,y)=\exp(-\gamma\lVert x-y\rVert^2)$: positive definite for any $\gamma>0$ (so $K+aI$ stays positive definite and every step above holds), universal (its infinite feature map approximates any smooth function), and with a single shape knob $\gamma$, the bandwidth. Because the exponent argument $\gamma\lVert x-y\rVert^2$ grows like the feature count for standardized inputs, I default $\gamma=1/n_{\text{features}}$ to keep it order $1$ regardless of dimension — otherwise in high dimension the kernel saturates to the identity and the fit learns nothing.

The one place this diverges from the bare recipe is the feature map, and it is what I actually fill into the scaffold. The descriptors span enormous dynamic ranges — `num_characters` in the billions, `vocab_size` in the thousands, `lr` near $10^{-3}$ — and the RBF uses Euclidean distance, so a raw representation would let the big-magnitude axis dictate every kernel value. Since the relationships are power-law, the natural representation is logarithmic, so I build a mixed map: standardized *raw* numerics **and** standardized $\log(1+\cdot)$ numerics (the log captures the power-law geometry, the raw captures any residual linear trend), concatenated with a one-hot encoding of the categorical `group`. The one-hot is what lets a single shared regressor still separate families inside the kernel — two runs from different groups are pushed apart in feature space, so the RBF similarity respects group membership without my fitting a separate model per group. I take a small $a=0.05$ (enough to keep $K+aI$ well-conditioned and damp noise, not so large it erases the signal) and the dimension-normalized $\gamma$. Crucially the target is used *raw* — no log transform on $y$ — because the vocab target is a unigram-normalised loss that can be negative, and I refuse to special-case it: a single black-box path that works whether the target is signed or not.

I expect this to be the *weakest* real rung, and the reason is structural, not a tuning miss. A smooth interpolator cannot extrapolate past the convex hull of its training inputs. The kernel value $\exp(-\gamma\lVert x-x_t\rVert^2)$ decays to zero as the query $x$ moves far from every training point, so for a held-out configuration larger and denser than anything in the fit — precisely the test region — every $k(x)_t\to 0$, the prediction $\sum_t c_t K(x_t,x)\to 0$, and the model collapses toward zero with no notion that loss should approach an irreducible floor along a power-law tail. It learns the *interior* smoothly and has nothing to say about the *boundary*, which is the only thing being tested. That predicted off-hull failure is the whole point of running it: it isolates the missing ingredient as the *right asymptotic form*, not more flexibility, and tells the next rung to stop being model-free and start imposing the power-law-plus-floor structure the literature laws carry.

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
