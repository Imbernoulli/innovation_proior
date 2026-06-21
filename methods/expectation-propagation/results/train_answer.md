The problem is to approximate a posterior that factorizes into a tractable base term and many local factors, `p(x | D) ∝ p0(x) ∏_i f_i(x)`. Even when each factor has a simple meaning, the product itself is usually intractable: a single mixture likelihood or nonconjugate observation can make the exact belief state exponentially complex or an arbitrary function. Existing ideas each leave something important behind. Laplace approximation trusts one mode and its curvature, so it can ignore mass away from that mode and underestimates uncertainty. Mean-field variational inference optimizes a global `KL(q || p)` objective, which often drives `q` away from regions the approximating family cannot represent well. Assumed-density filtering keeps the local factorization and projects after each factor, but it is one-pass and order-dependent: information discarded early cannot be reconsidered once later evidence arrives. Loopy belief propagation enforces local consistency, yet its messages are exact functions of variables, which is too rich for hybrid or continuous nonconjugate models.

What is needed is an iterative, local procedure that preserves selected moments, revisits old decisions, and works when exact messages are impossible to store. The right move is to keep every exact factor except the base as a separate tractable approximation, then repeatedly test and repair each local approximation against its true factor in the context of all the others.

The method is Expectation Propagation (EP). EP maintains a global approximation `q(x) ∝ p0(x) ∏_i t_i(x)`, where each site `t_i(x)` is a tractable replacement for the corresponding exact factor `f_i(x)`. The family for `q` and the sites is chosen from an exponential family, most commonly a Gaussian. The algorithm cycles through the sites and performs four steps for each. First, delete the current site to form the cavity distribution `q_{-i}(x) ∝ q(x) / t_i(x)`; this is the belief supplied by every other factor. Second, insert the exact factor into the cavity to form the tilted distribution `p̂_i(x) ∝ f_i(x) q_{-i}(x)`. This tilted object is not tractable, but it is locally exact for factor `i` and usually much cheaper than the full posterior. Third, project the tilted distribution back into the tractable family by minimizing `KL(p̂_i || q_new)`, which for an exponential family reduces to moment matching on the retained sufficient statistics. Fourth, replace the site by the ratio `t_i_new(x) ∝ q_new(x) / q_{-i}(x)`, which in natural parameters is just subtraction of the cavity parameters from the projected parameters. The change can be damped to improve convergence.

The fixed point has a clear meaning: for every site `i`, deleting it, inserting the exact factor, and projecting recovers the same retained moments as the current global `q`. The approximation is therefore not a single projection onto an intractable target, but the simultaneous solution of many local consistency tests. When the approximation family is fully factorized and the sites are messages into individual variables, EP reduces to loopy belief propagation, but EP generalizes this by allowing richer compressed messages such as correlated Gaussians. EP is especially useful when each tilted update is manageable, for example a non-Gaussian likelihood multiplied by a Gaussian cavity. It is not a convex guarantee: fixed points need not be unique, iterations can oscillate or diverge, and a poor family can hide multimodality. In Gaussian implementations site precisions can become negative, so practical code must check cavities, skip bad updates, or restrict site variances.

The approximate evidence is the normalizing constant of `p0(x) ∏_i t_i(x)` after accounting for the site log-scales. In the common Gaussian case this combines the sum of site natural-parameter terms with prior and posterior determinants. EP extends assumed-density filtering by making the one-pass projection reversible, and it extends loopy belief propagation by compressing messages into exponential-family forms that can handle continuous nonconjugate models.

```python
import numpy as np
from scipy.stats import norm


def ep_probit_1d(y, prior_var=1.0, max_iter=100, tol=1e-6, damping=0.5):
    """
    Gaussian Expectation Propagation for a 1D probit model:
        p(x) ∝ N(x; 0, prior_var) * ∏_i Φ(y_i * x)
    Each site t_i(x) is represented by natural parameters
    (lambda_i, beta_i) so that t_i(x) ∝ exp(beta_i * x - 0.5 * lambda_i * x**2).
    """
    y = np.asarray(y, dtype=float)
    n = y.size

    site_lambda = np.zeros(n)
    site_beta = np.zeros(n)

    post_lambda = 1.0 / prior_var
    post_beta = 0.0

    for _ in range(max_iter):
        post_beta_old = post_beta

        for i in range(n):
            # 1. Cavity deletion: remove site i from the posterior.
            cav_lambda = post_lambda - site_lambda[i]
            cav_beta = post_beta - site_beta[i]
            if cav_lambda <= 0.0:
                continue
            cav_var = 1.0 / cav_lambda
            cav_mean = cav_beta * cav_var

            # 2. Exact local insertion with probit likelihood f_i(x) = Φ(y_i * x).
            s = np.sqrt(1.0 + cav_var)
            alpha = y[i] * cav_mean / s
            z = norm.cdf(alpha)
            if z < 1e-12:
                continue
            ratio = norm.pdf(alpha) / z

            # 3. Moment projection: match mean and variance of the tilted distribution.
            tilted_mean = cav_mean + y[i] * cav_var * ratio / s
            tilted_var = cav_var - (cav_var**2 * ratio / s**2) * (alpha + ratio)
            if tilted_var <= 0.0:
                continue

            new_post_lambda = 1.0 / tilted_var
            new_post_beta = tilted_mean / tilted_var

            # 4. Site replacement in natural parameters.
            new_site_lambda = new_post_lambda - cav_lambda
            new_site_beta = new_post_beta - cav_beta

            site_lambda[i] = (1.0 - damping) * site_lambda[i] + damping * new_site_lambda
            site_beta[i] = (1.0 - damping) * site_beta[i] + damping * new_site_beta

            # Recompute global posterior natural parameters.
            post_lambda = 1.0 / prior_var + site_lambda.sum()
            post_beta = site_beta.sum()

        if abs(post_beta - post_beta_old) < tol:
            break

    post_var = 1.0 / post_lambda
    post_mean = post_beta * post_var
    return post_mean, post_var


# Example: 1D probit classification with 50 binary labels.
np.random.seed(0)
n = 50
x_true = 0.7
y = np.where(np.random.randn(n) + x_true > 0, 1, -1)
mean, var = ep_probit_1d(y)
print(f"EP posterior mean: {mean:.3f}, std: {np.sqrt(var):.3f}")
```
