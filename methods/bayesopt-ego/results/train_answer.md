We need to minimize an expensive, deterministic, gradient-free black-box function y(x) over a box in design space, using as few evaluations as possible because each call might take hours. Grid search, random search, and multistart local methods are immediately ruled out: the dimensionality curse makes the former infeasible, and local methods on multimodal surfaces consume the whole budget just to find a nearby basin while ignoring the information already gathered. Even naive surrogate methods fall short. A classical response-surface fit assumes independent errors and gives only a global variance, so it cannot say where it is ignorant; pure exploitation of the fitted surface simply refines the nearest local minimum, while pure uncertainty-based exploration wastes the budget mapping unimportant regions.

The method that solves this is Efficient Global Optimization, abbreviated EGO. It was introduced by Jones, Schonlau and Welch in 1998 as a way to combine a kriging surrogate with an acquisition function that balances exploitation and exploration automatically. The core observation is that for a deterministic computer code the residuals are not noise but a correlated continuous function of x, so the right model is a Gaussian random field rather than independent-error regression. EGO builds that surrogate and then repeatedly asks it where to sample next by maximizing the expected improvement over the current best value.

The surrogate is the DACE or kriging model: y(x_i) = mu + epsilon(x_i), where epsilon is a zero-mean Gaussian process with variance sigma^2 and correlation Corr(epsilon(x_i), epsilon(x_j)) = exp(- sum_h theta_h |x_{ih} - x_{jh}|^{p_h}). The parameters theta_h encode per-variable activity and p_h in [1,2] encode smoothness. Given fixed correlation parameters, mu and sigma^2 have closed-form maximum-likelihood estimates, so only the 2k correlation parameters need to be optimized numerically. The predictor y_hat(x*) = mu_hat + r' R^{-1} (y - 1 mu_hat) interpolates the observed data, and its mean-squared error s^2(x*) is zero at sampled points and rises to about sigma^2 far away. This gives the two quantities needed to drive search: a predicted mean to exploit and a local uncertainty to explore.

The acquisition criterion is Expected Improvement. Treat the unknown value at x as Y(x) ~ Normal(y_hat(x), s^2(x)) and let f_min be the best value seen so far. The improvement random variable is I(x) = max(f_min - Y(x), 0). Rather than maximizing only the probability of improvement, which ignores the magnitude and clusters around the incumbent, EGO maximizes the expected improvement E[I(x)]. With z = (f_min - y_hat)/s this has the closed form EI(x) = (f_min - y_hat) Phi(z) + s phi(z), where Phi and phi are the standard normal cdf and pdf. The first term rewards exploitation, the second rewards exploration, and no hand-tuned trade-off parameter is required. EI is zero at evaluated points, positive elsewhere, and highly multimodal, so iterating it produces a genuine global search. It is monotone decreasing in y_hat and increasing in s, which also makes it amenable to branch-and-bound maximization, and its maximum value supplies a natural stopping rule.

The algorithm is straightforward. Start with a space-filling Latin hypercube design so that low-dimensional projections are well covered and the correlation matrix is not immediately singular. Evaluate the expensive objective on those points, fit the kriging model by maximum likelihood, and check cross-validated standardized residuals. If the diagnostics are poor, try a response transformation such as log y or -1/y. Then repeatedly maximize EI over the box. If the maximum EI is below a small fraction of |f_min|, stop; otherwise evaluate the objective at the EI maximizer, append the point, refit, and repeat. In practice the correlation matrix can become nearly singular when the surface is very smooth or when points cluster late in the run, so a tiny numerical nugget or SVD-based inversion should be used.

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def latin_hypercube(n_points, bounds, rng):
    bounds = np.asarray(bounds, float)
    k = len(bounds)
    cut = np.linspace(0.0, 1.0, n_points + 1)
    u = rng.uniform(size=(n_points, k))
    pts = cut[:n_points, None] + u * (cut[1] - cut[0])
    for j in range(k):
        pts[:, j] = rng.permutation(pts[:, j])
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + pts * (hi - lo)


class CorrelatedSurrogate:
    """GP/kriging surrogate with ARD length scales and a tiny numerical nugget."""

    def __init__(self, alpha=1e-10):
        self.alpha = alpha
        self.gp = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        kernel = (ConstantKernel(1.0) *
                  Matern(length_scale=np.ones(X.shape[1]), nu=2.5))
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=self.alpha, normalize_y=True,
            n_restarts_optimizer=10)
        self.gp.fit(X, y)
        return self

    def predict(self, X, return_std=True):
        return self.gp.predict(np.atleast_2d(X), return_std=return_std)


def expected_improvement(X, surrogate, f_min, xi=0.0):
    mu, std = surrogate.predict(X, return_std=True)
    mu = np.atleast_1d(mu)
    std = np.atleast_1d(std)
    ei = np.zeros_like(mu)
    mask = std > 1e-12
    improve = f_min - xi - mu[mask]
    z = improve / std[mask]
    ei[mask] = improve * norm.cdf(z) + std[mask] * norm.pdf(z)
    return ei


def maximize_acquisition(acq_fn, bounds, rng, n_restarts=20, n_raw=10000):
    bounds = np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    raw = lo + rng.uniform(size=(n_raw, len(bounds))) * (hi - lo)
    vals = acq_fn(raw)
    seeds = raw[np.argsort(vals)[-n_restarts:]]
    best_x, best_val = raw[vals.argmax()], float(vals.max())
    for x0 in seeds:
        res = minimize(lambda x: -float(acq_fn(x)[0]),
                       x0, bounds=list(map(tuple, bounds)), method="L-BFGS-B")
        if -res.fun > best_val:
            best_x, best_val = res.x, -res.fun
    return best_x, best_val


def efficient_global_optimization(objective, bounds, n_init=10, max_evals=40,
                                  ei_tol_frac=0.01, xi=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = latin_hypercube(n_init, bounds, rng)
    y = np.array([objective(x) for x in X])
    surrogate = CorrelatedSurrogate()
    for _ in range(max_evals - n_init):
        surrogate.fit(X, y)
        f_min = y.min()
        acq_fn = lambda Xcand: expected_improvement(Xcand, surrogate, f_min, xi)
        x_next, ei = maximize_acquisition(acq_fn, bounds, rng)
        if ei < ei_tol_frac * max(abs(f_min), 1e-12):
            break
        y_next = objective(x_next)
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
    i = y.argmin()
    return X[i], y[i]
```
