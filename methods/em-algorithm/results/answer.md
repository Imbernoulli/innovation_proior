# The EM algorithm

## Problem

Maximize the observed-data log-likelihood $L(\phi)=\log g(y\mid\phi)$ when the data are *incomplete*: $y$ is a known many-to-one image $x\mapsto y(x)$ of complete data $x$ drawn from $f(x\mid\phi)$, so

$$g(y\mid\phi)=\int_{\mathcal X(y)} f(x\mid\phi)\,dx.$$

Direct maximization fails because the marginalization over the hidden coordinates buries them inside a logarithm (for a mixture, $L=\sum_i\log\sum_k \pi_k p_k(y_i)$ — a sum of logs of sums, with no closed-form score equations).

## Key idea

The complete-data log-likelihood $\log f(x\mid\phi)$ is easy to maximize; the hard part is only that $x$ is unobserved. So work with its conditional expectation. Define the conditional density of the complete data given the observed data, $k(x\mid y,\phi)=f(x\mid\phi)/g(y\mid\phi)$ (the posterior over the hidden coordinates), and the **$Q$-function**

$$Q(\phi\mid\phi')=E\big[\log f(x\mid\phi)\,\big|\,y,\phi'\big].$$

Iterate:

> **E-step:** compute $Q(\phi\mid\phi^{(p)})$ — i.e. fill in the hidden coordinates by their posterior under the current $\phi^{(p)}$.
> **M-step:** set $\phi^{(p+1)}=\arg\max_\phi Q(\phi\mid\phi^{(p)})$.

## The monotone-likelihood theorem

Write $H(\phi\mid\phi')=E[\log k(x\mid y,\phi)\mid y,\phi']$. From $\log f=\log g+\log k$ and taking $E[\cdot\mid y,\phi']$,

$$L(\phi)=Q(\phi\mid\phi')-H(\phi\mid\phi')\qquad\text{for every }\phi'. \tag{1}$$

**Lemma (Jensen / KL $\ge 0$).** For all $\phi$, $H(\phi\mid\phi')\le H(\phi'\mid\phi')$, with equality iff $k(\cdot\mid y,\phi)=k(\cdot\mid y,\phi')$ a.e.

*Proof.* $H(\phi\mid\phi')-H(\phi'\mid\phi')=E\!\big[\log\frac{k(x\mid y,\phi)}{k(x\mid y,\phi')}\,\big|\,y,\phi'\big]\le \log E\!\big[\frac{k(x\mid y,\phi)}{k(x\mid y,\phi')}\,\big|\,y,\phi'\big]=\log\!\int k(x\mid y,\phi)\,dx=\log 1=0$, by concavity of $\log$ (Jensen); equivalently the left side equals $-\mathrm{KL}\big(k(\cdot\mid y,\phi')\,\|\,k(\cdot\mid y,\phi)\big)\le 0$. $\square$

**Theorem (monotone ascent).** Any update with $Q(\phi^{(p+1)}\mid\phi^{(p)})\ge Q(\phi^{(p)}\mid\phi^{(p)})$ (a *generalized* EM step; the M-step's argmax is one such) satisfies $L(\phi^{(p+1)})\ge L(\phi^{(p)})$.

*Proof.* Subtracting (1) at $\phi^{(p+1)}$ and at $\phi^{(p)}$ (both with $\phi'=\phi^{(p)}$),

$$L(\phi^{(p+1)})-L(\phi^{(p)})=\underbrace{\big[Q(\phi^{(p+1)}\mid\phi^{(p)})-Q(\phi^{(p)}\mid\phi^{(p)})\big]}_{\ge\,0\ \text{(M-step)}}+\underbrace{\big[H(\phi^{(p)}\mid\phi^{(p)})-H(\phi^{(p+1)}\mid\phi^{(p)})\big]}_{\ge\,0\ \text{(Lemma)}}\ \ge 0.$$

Equality requires both brackets zero; at a fixed point $\phi^\ast$, $\nabla_\phi Q(\cdot\mid\phi^\ast)|_{\phi^\ast}=0$ and $\nabla_\phi H(\cdot\mid\phi^\ast)|_{\phi^\ast}=0$, so $\nabla L(\phi^\ast)=0$: fixed points are stationary points of $L$ (not necessarily global maxima). $\square$

## Variational lower-bound view

For any distribution $q$ over the hidden coordinates, with entropy $\mathcal H(q)=-E_q[\log q]$,

$$\log g(y\mid\phi)=\underbrace{E_q[\log f(x\mid\phi)]+\mathcal H(q)}_{F(q,\phi)}+\mathrm{KL}\big(q\,\big\|\,k(\cdot\mid y,\phi)\big).$$

Since $\mathrm{KL}\ge 0$, $F(q,\phi)$ is a lower bound on $L(\phi)$, **tight** when $q=k(\cdot\mid y,\phi)$. EM is **coordinate ascent on $F$**: the E-step maximizes $F$ over $q$ (set $q$ to the posterior, closing the gap), the M-step maximizes $F$ over $\phi$ (maximize $Q$, since $\mathcal H(q)$ is $\phi$-free). Monotonicity is then immediate: $L(\phi^{(p+1)})\ge F(q^{(p)},\phi^{(p+1)})\ge F(q^{(p)},\phi^{(p)})=L(\phi^{(p)})$. The same monotonicity proof covers a generalized M-step whenever it increases $Q$.

## Worked special case: Gaussian mixture

Complete data $x=(y,z)$ with one-hot labels $z_i$; $\log f$ is linear in $z_{ik}$, so the E-step needs only the posterior responsibilities $r_{ik}=E[z_{ik}\mid y_i,\phi^{(p)}]$, and the M-step is a responsibility-weighted complete-data fit.

```python
import numpy as np

def logsumexp(a, axis=1, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)

def estimate_log_gaussian_prob_full(X, means, covariances):
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    log_prob = np.empty((n_samples, n_components), dtype=X.dtype)
    for k in range(n_components):
        chol = np.linalg.cholesky(covariances[k])
        diff = X - means[k]
        whitened = np.linalg.solve(chol, diff.T).T
        mahalanobis = np.sum(whitened * whitened, axis=1)
        log_det = np.sum(np.log(np.diag(chol)))
        log_prob[:, k] = -0.5 * (
            n_features * np.log(2.0 * np.pi) + mahalanobis
        ) - log_det
    return log_prob

def estimate_log_prob_resp(X, weights, means, covariances):
    weighted_log_prob = estimate_log_gaussian_prob_full(X, means, covariances)
    weighted_log_prob = weighted_log_prob + np.log(weights)
    log_prob_norm = logsumexp(weighted_log_prob, axis=1, keepdims=True)
    log_resp = weighted_log_prob - log_prob_norm
    return np.squeeze(log_prob_norm, axis=1), log_resp

def estimate_gaussian_parameters_full(X, resp, reg_covar=1e-6):
    n_components = resp.shape[1]
    n_features = X.shape[1]
    nk = resp.sum(axis=0) + 10.0 * np.finfo(resp.dtype).eps
    means = (resp.T @ X) / nk[:, None]
    covariances = np.empty((n_components, n_features, n_features), dtype=X.dtype)
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = ((resp[:, k] * diff.T) @ diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
    return nk, means, covariances

def m_step_full(X, log_resp, reg_covar=1e-6):
    resp = np.exp(log_resp)
    weights, means, covariances = estimate_gaussian_parameters_full(
        X, resp, reg_covar=reg_covar
    )
    weights = weights / weights.sum()
    return weights, means, covariances

def em_gmm_full(X, weights, means, covariances, max_iter=100, tol=1e-3, reg_covar=1e-6):
    lower_bounds = []
    lower_bound = -np.inf
    for _ in range(max_iter):
        previous = lower_bound
        log_prob_norm, log_resp = estimate_log_prob_resp(X, weights, means, covariances)
        weights, means, covariances = m_step_full(X, log_resp, reg_covar=reg_covar)
        lower_bound = float(np.mean(log_prob_norm))
        lower_bounds.append(lower_bound)
        if np.isfinite(previous) and abs(lower_bound - previous) < tol:
            break
    return weights, means, covariances, lower_bounds
```

The E-step turns the intractable $\sum_i\log\sum_k\pi_k\mathcal N(y_i\mid\mu_k,\Sigma_k)$ into $Q=\sum_{i,k} r_{ik}[\log\pi_k+\log\mathcal N(y_i\mid\mu_k,\Sigma_k)]$ — the log moves inside the sum, the $R$ components decouple — and the M-step solves each in closed form. For the weights, maximize $\sum_k N_k\log\pi_k$ with $\sum_k\pi_k=1$: $N_k/\pi_k+\lambda=0$, so $\pi_k=-N_k/\lambda$, and summing gives $\lambda=-n$ and $\pi_k=N_k/n$. The Gaussian updates are $\mu_k=\sum_i r_{ik}y_i/N_k$ and $\Sigma_k=\sum_i r_{ik}(y_i-\mu_k)(y_i-\mu_k)^{\mathsf T}/N_k$. The code keeps the E-step in log space, exponentiates `log_resp` only for the weighted M-step, adds `reg_covar` to each covariance diagonal, and normalizes the weights, matching the Gaussian-mixture implementation pattern. By the theorem, the observed-data log-likelihood does not decrease across any E-M sweep.
