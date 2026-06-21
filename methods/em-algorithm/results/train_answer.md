Maximum likelihood estimation is straightforward when we observe the complete data that our model is written for: we take logs, differentiate, set derivatives to zero, and solve. The difficulty is that real measurements are often incomplete. We may see grouped or censored counts, missing entries, latent component labels, or any other many-to-one summary of a richer underlying record. In every such case the likelihood we actually want to maximize is the marginal likelihood over all complete records consistent with what we observed. The logarithm sits outside the integral or sum, so the neat sufficient-statistic equations that made the complete-data problem easy no longer apply.

A naive response is to fill in the missing values with the best current guess and then optimize as if the filled-in data were real. That is fast, but it silently changes the objective. A single imputation is not the same as integrating over uncertainty, and a method built on hard fill-in gives no reason to believe that the observed likelihood is improving. What is needed is a way to keep the observed likelihood as the true target while still exploiting the tractability of the complete-data problem.

The right method is the Expectation-Maximization algorithm, or EM. It starts from the same pair of densities: f(x | phi) for the complete data and g(y | phi) for the observed data, where g is obtained by integrating f over all complete values x compatible with y. The conditional density of the complete data given the observation is k(x | y, phi) = f(x | phi) / g(y | phi). At a current parameter phi_p, EM forms the expected complete-data log likelihood under this conditional distribution: Q(phi | phi_p) = E[log f(X | phi) | y, phi_p]. The key identity is that, on the set of complete data compatible with y, log f(x | phi) = log g(y | phi) + log k(x | y, phi). Taking expectations under k(. | y, phi_p) gives L(phi) = Q(phi | phi_p) - H(phi | phi_p), where L is the observed log likelihood and H is the expected log conditional density.

The update has two parts. The E-step computes the conditional distribution of the missing data given the observations under the current parameter and uses it to build Q. The M-step chooses a new parameter phi_{p+1} that makes Q larger; exact maximization is common, but any increase is enough for the ascent property. The proof that this raises the observed likelihood follows from Jensen's inequality applied to H. Because H(phi | phi_p) is never larger than H(phi_p | phi_p), the gain in Q dominates any movement in H. Therefore increasing Q also increases L, unless the conditional distributions already agree. For regular exponential-family complete-data models, the E-step reduces to computing expected sufficient statistics, and the M-step is exactly the usual closed-form maximum-likelihood update with those expectations in place of the unobserved statistics.

A canonical example is a Gaussian mixture. The complete data add a hidden one-hot component label to each observed vector. The E-step computes, for each point and each component, the posterior probability that the point came from that component. These responsibilities are then treated as fractional membership counts in the M-step: each component's weight is the normalized sum of its responsibilities, its mean is the responsibility-weighted average of the data, and its covariance is the responsibility-weighted scatter. The code below implements this version directly. It avoids external machine-learning libraries so the logic is transparent, but it follows the same weighted-sufficient-statistic structure used in production packages.

```python
import numpy as np

class GaussianMixtureEM:
    def __init__(self, n_components, max_iter=100, tol=1e-5, seed=0):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)

    def _initialize(self, X):
        n, d = X.shape
        self.weights_ = np.ones(self.K) / self.K
        idx = self.rng.choice(n, self.K, replace=False)
        self.means_ = X[idx].copy()
        self.covariances_ = np.array([
            np.cov(X.T) + 1e-6 * np.eye(d) for _ in range(self.K)
        ])

    def _gaussian_pdf(self, X, mean, cov):
        d = X.shape[1]
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov = cov + 1e-6 * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov)
        diff = X - mean
        inv_cov = np.linalg.inv(cov)
        mahalanobis = np.sum(diff @ inv_cov * diff, axis=1)
        log_prob = -0.5 * (d * np.log(2 * np.pi) + logdet + mahalanobis)
        return np.exp(log_prob)

    def _e_step(self, X):
        n = X.shape[0]
        log_resp = np.zeros((n, self.K))
        for k in range(self.K):
            log_resp[:, k] = (
                np.log(self.weights_[k] + 1e-16)
                + np.log(self._gaussian_pdf(X, self.means_[k], self.covariances_[k]) + 1e-16)
            )
        log_resp -= log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def _m_step(self, X, resp):
        n, d = X.shape
        counts = resp.sum(axis=0) + 1e-16
        self.weights_ = counts / n
        for k in range(self.K):
            self.means_[k] = (resp[:, k:k+1] * X).sum(axis=0) / counts[k]
            diff = X - self.means_[k]
            self.covariances_[k] = (
                (resp[:, k:k+1] * diff).T @ diff / counts[k]
                + 1e-6 * np.eye(d)
            )

    def _log_likelihood(self, X):
        n = X.shape[0]
        ll = np.zeros((n, self.K))
        for k in range(self.K):
            ll[:, k] = self.weights_[k] * self._gaussian_pdf(
                X, self.means_[k], self.covariances_[k]
            )
        return np.log(ll.sum(axis=1) + 1e-16).sum()

    def fit(self, X):
        self._initialize(X)
        prev_ll = -np.inf
        for _ in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            ll = self._log_likelihood(X)
            if ll - prev_ll < self.tol:
                break
            prev_ll = ll
        return self

    def predict(self, X):
        resp = self._e_step(X)
        return resp.argmax(axis=1)


def example():
    rng = np.random.default_rng(0)
    n = 300
    X = np.vstack([
        rng.multivariate_normal([0.0, 0.0], [[0.5, 0.0], [0.0, 0.5]], n),
        rng.multivariate_normal([5.0, 5.0], [[0.8, 0.0], [0.0, 0.8]], n),
    ])
    model = GaussianMixtureEM(n_components=2, max_iter=100, seed=0)
    model.fit(X)
    print("weights:", model.weights_)
    print("means:\n", model.means_)


if __name__ == "__main__":
    example()
```

EM is not a global optimizer. It guarantees monotone ascent of the observed likelihood toward a local maximum or stationary point, and its practical speed depends on how much information is lost in the move from complete data to observed data. When the missing information is large, the posterior expectations change slowly and convergence can be patience-testing. Nevertheless, by keeping the observed likelihood as the true objective and using the complete-data likelihood only as an auxiliary device, EM gives a principled and broadly applicable way to fit models with hidden structure.
