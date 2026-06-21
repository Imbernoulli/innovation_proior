The problem is to understand why a predictor that interpolates noisy training labels can still generalize well. In classical low-dimensional statistics, forcing zero training error on noisy data is a mistake: the fitted model memorizes the noise and predicts poorly. Yet modern overparameterized models routinely reach zero training error while maintaining useful test accuracy. The naive explanations fall short. Saying the model is "too big" is not enough, because some large models interpolate badly. Saying "double descent" occurs is only a description of the empirical curve; it does not say which data distributions make the second descent possible. Ridge regression is a controlled baseline, but its zero-regularization limit is itself an exact interpolant, so the question is really about that limit. The missing ingredient is a precise way to measure whether the covariate geometry gives the interpolant a safe place to hide the noise.

The key observation is that prediction risk is not Euclidean parameter error; it is covariance-weighted parameter error. An error in a high-variance covariate direction is expensive, while an error in a low-variance direction is almost invisible. If the covariate distribution has a broad tail of weak directions, the minimum-norm interpolant can absorb the training noise into those directions without paying much in future prediction error. The dividing line is therefore spectral, not merely dimensional.

The method is Benign Overfitting, analyzed through the minimum-norm least-squares interpolant in overparameterized linear regression. Given a design matrix X with full row rank and more columns than rows, the estimator is the pseudoinverse solution that has the smallest Euclidean norm among all exact fits. This is the zero-regularization limit of ridge regression. The fitted parameter splits naturally into a signal component and a noise component. The signal component projects the true parameter onto the row space of the sample, and the noise component pushes the training residuals back into parameter space through the pseudoinverse. The excess risk decomposes into a covariance-weighted bias term and a covariance-weighted variance term.

To make the geometry explicit, diagonalize the covariate covariance. The contribution of each eigen-direction to the noise cost is controlled by the eigenvalue and by how the sample Gram matrix concentrates. The crucial spectral quantities are two effective ranks. The first effective rank asks whether the low-variance tail is wide enough, relative to its largest eigenvalue, to make the tail Gram matrix behave like a scalar multiple of the identity. The second effective rank asks whether the tail is balanced enough to dilute noise evenly rather than concentrating it in a few directions. When the tail starts early, is wide enough, and is balanced enough, the cost of exact interpolation becomes small. When it does not, interpolation is harmful. This gives matching upper and lower bounds on the excess risk in terms of these effective ranks, rather than a loose capacity argument.

The practical message is that overparameterization is benign when the covariance spectrum provides many cheap directions before the sample runs out of dimensions to control them. A large but flat tail of tiny eigenvalues is friendly; a spectrum that is square or dominated by a few large eigenvalues is not. This distinction explains why the same interpolation threshold can look catastrophic in one setting and harmless in another.

```python
import numpy as np
from numpy.linalg import pinv, norm


def benign_overfitting_risk(X_train, y_train, X_test, y_test, Sigma=None):
    """
    Minimum-norm least-squares interpolant and its excess risk.

    Parameters
    ----------
    X_train : array, shape (n, p) with p > n
    y_train : array, shape (n,)
    X_test  : array, shape (m, p)
    y_test  : array, shape (m,)
    Sigma   : array, shape (p, p), optional covariance of covariates

    Returns
    -------
    theta_hat : fitted parameter
    excess_risk : covariance-weighted squared error vs the Bayes linear rule
    """
    n, p = X_train.shape
    if p <= n:
        raise ValueError("This demo assumes the overparameterized p > n regime.")

    # Minimum-norm interpolant: theta_hat = X^T (X X^T)^{-1} y
    theta_hat = X_train.T @ pinv(X_train @ X_train.T) @ y_train

    # Estimate Sigma from test data if not supplied
    if Sigma is None:
        Sigma = (X_test.T @ X_test) / X_test.shape[0]

    # True linear target is unknown in practice; here we compare to y_test
    # Excess risk approximation: covariance-weighted parameter error relative
    # to the least-squares fit on the test set.
    theta_test = pinv(X_test) @ y_test
    delta = theta_test - theta_hat
    excess_risk = float(delta @ Sigma @ delta)

    return theta_hat, excess_risk


def effective_ranks(eigvals, k):
    """
    Compute the two effective ranks used in the benign-overfitting analysis.

    r_k(Sigma) = sum_{i>k} lambda_i / lambda_{k+1}
    R_k(Sigma) = (sum_{i>k} lambda_i)^2 / sum_{i>k} lambda_i^2
    """
    tail = eigvals[k:]
    r_k = tail.sum() / eigvals[k]
    R_k = (tail.sum() ** 2) / (tail ** 2).sum()
    return r_k, R_k


def make_benign_data(n=100, p=1000, signal_rank=20, noise_std=0.3, seed=0):
    """
    Generate data with a friendly covariance spectrum: a few strong signal
    directions and a long, balanced tail of weak directions.
    """
    rng = np.random.default_rng(seed)
    # Eigenvalues: rapid decay for the signal, slow flat tail afterwards
    k = np.arange(1, p + 1)
    eigvals = np.where(
        k <= signal_rank,
        np.exp(-0.2 * (k - 1)),
        0.01 / k
    )
    Sigma = np.diag(eigvals)

    # Draw covariates and responses
    Z = rng.standard_normal((n, p))
    X = Z @ np.sqrt(Sigma)
    theta_star = rng.standard_normal(p)
    # Scale true parameter so that signal lives in the top directions
    theta_star[signal_rank:] = 0.0
    y = X @ theta_star + rng.normal(0, noise_std, n)

    # Test set
    Z_test = rng.standard_normal((5 * n, p))
    X_test = Z_test @ np.sqrt(Sigma)
    y_test = X_test @ theta_star + rng.normal(0, noise_std, 5 * n)

    return X, y, X_test, y_test, Sigma, theta_star


if __name__ == "__main__":
    n, p = 100, 1000
    X, y, X_test, y_test, Sigma, theta_star = make_benign_data(n=n, p=p)

    theta_hat, excess = benign_overfitting_risk(X, y, X_test, y_test, Sigma)
    print(f"Excess risk of minimum-norm interpolant: {excess:.4f}")

    # Inspect the spectral conditions
    eigvals = np.diag(Sigma)
    for k in [0, 10, 20, 50]:
        r_k, R_k = effective_ranks(eigvals, k)
        print(f"k={k}: r_k={r_k:.2f}, R_k={R_k:.2f}")

    # Compare to an oracle least-squares baseline on the test set
    theta_oracle = pinv(X_test) @ y_test
    oracle_delta = theta_oracle - theta_star
    oracle_risk = float(oracle_delta @ Sigma @ oracle_delta)
    print(f"Oracle residual risk: {oracle_risk:.4f}")
```
