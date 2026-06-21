The model in front of me is the most innocent thing imaginable: $y = X\beta + \varepsilon$ with an $n \times p$ design matrix $X$, an unknown coefficient vector $\beta$, and noise with $\mathbb{E}[\varepsilon] = 0$, $\mathrm{Cov}(\varepsilon) = \sigma^2 I$. I want to estimate $\beta$, and the two-hundred-year-old recipe is ordinary least squares: minimize the residual sum of squares $\|y - X\beta\|^2$, set the gradient to zero, get the normal equations $X'X\beta = X'y$, and read off $\hat\beta = (X'X)^{-1}X'y$ when $X'X$ inverts. That estimator is unbiased, it is the maximum-likelihood estimate under Gaussian noise, and the Gauss-Markov theorem crowns it as the best linear unbiased estimator there is. On a clean, orthogonal design I would be done. The trouble is the data I actually have, where the columns of $X$ are correlated, nearly redundant, sometimes more numerous than the rows — and there least squares falls apart in a way the optimality theorem completely fails to warn me about.

The shape of the failure tells me the shape of the cure. The estimator's quality lives in its covariance $\mathrm{Cov}(\hat\beta) = \sigma^2 (X'X)^{-1}$, so diagonalize $X'X = \sum_j d_j^2 v_j v_j'$, where the $d_j$ are the singular values of $X$ (if $X = UDV'$ then $X'X = VD'DV'$) and the $v_j$ the right singular vectors. Inverting a symmetric matrix inverts its eigenvalues on the same eigenvectors, so

$$\mathrm{Cov}(\hat\beta) = \sigma^2 (X'X)^{-1} = \sigma^2 \sum_j d_j^{-2}\, v_j v_j'.$$

The variance along direction $v_j$ is $\sigma^2 / d_j^2$, and a small singular value is exactly what collinearity is: if two covariates are nearly proportional, there is a combination $v_j$ along which $X$ barely moves, so $d_j$ is tiny, the data carries almost no information about $\beta$ there, and the noise fills the vacuum and gets amplified by $1/d_j^2$. That is why the coefficients come out with implausible magnitudes and signs that flip on a one-point perturbation. Push it to the limit — exact dependence, or $p > n$ — and some $d_j = 0$, the inverse does not exist, the normal equations have a whole affine space of solutions, and least squares does not even return an answer. The obvious fixes do not satisfy me. Dropping collinear covariates by subset selection, or projecting onto the top principal components, does kill the $1/d_j^2$ blow-up, but each is a guillotine: a direction is kept whole or severed whole, so the estimator is a discontinuous, high-variance function of where the threshold falls, and principal-component regression deletes directions by their variance in $X$ alone, throwing out a low-$d_j$ direction even when it is genuinely predictive of $y$. The minimum-norm pseudo-inverse merely restores a unique answer without damping the noise amplification at all. I want a dial, not a switch — something that damps the unstable directions smoothly, in proportion to how unstable they are, while barely touching the healthy ones.

I propose ridge regression. The smallest honest move against the $d_j^{-2}$ explosion is to replace $d_j^2$ by $d_j^2 + \lambda$ before inverting: for the big-$d_j$ directions the floor $\lambda$ is negligible and nothing changes, but the sick small-$d_j$ directions can no longer fall below it, so their variance is capped instead of diverging. Adding $\lambda$ to every eigenvalue is, in matrix terms, replacing $X'X$ by $X'X + \lambda I$, which gives

$$\hat\beta(\lambda) = (X'X + \lambda I)^{-1} X'y.$$

Because $X'X$ is positive semidefinite and $\lambda I$ is positive definite for $\lambda > 0$, their sum is positive definite with every eigenvalue at least $\lambda$ — so it is always invertible, even when $X'X$ was singular, even when $p > n$. This looked at first like an ad-hoc patch on the linear algebra, so I reverse-engineered what objective it optimizes. Charge the fit for large coefficients: $L(\beta) = \|y - X\beta\|^2 + \lambda\|\beta\|^2$. Its gradient is $-2X'(y - X\beta) + 2\lambda\beta = -2X'y + 2(X'X + \lambda I)\beta$, and setting it to zero gives back exactly $(X'X + \lambda I)\beta = X'y$. So the diagonal patch is precisely the minimizer of L2-penalized least squares, and it is a genuine global minimum, not a stationary point: the Hessian $2(X'X + \lambda I)$ is positive definite, so $L$ is strictly convex with a unique minimizer. That reframes $\lambda$ as the penalty strength, the exchange rate between fitting and shrinking — at $\lambda = 0$ I recover OLS, as $\lambda \to \infty$ the estimate shrinks to $0$. By Lagrangian duality the same objective is $\min \|y - X\beta\|^2$ subject to a budget $\|\beta\|^2 \le c$, with $\lambda$ the KKT multiplier, so geometrically the ellipsoidal residual contours are forced to meet an L2 ball around the origin and the solution is a shrunk OLS estimate.

The shrinkage is cleanest in the singular value decomposition. Pushing $X = UDV'$ through, $\hat\beta(\lambda) = (VD'DV' + \lambda VV')^{-1} VD'U'y = V(D'D + \lambda I)^{-1}D'U'y$, so the coordinate that was $d_j^{-1}(u_j'y)$ under OLS becomes $d_j(d_j^2 + \lambda)^{-1}(u_j'y)$ — each direction multiplied by the factor

$$\frac{d_j^2}{d_j^2 + \lambda} \in (0, 1].$$

For a large $d_j$ this is near $1$ (barely touched); for a tiny $d_j$ it is near $d_j^2/\lambda \approx 0$ (shrunk almost to nothing). The estimator damps each direction in exact proportion to its instability, and the trace of the hat matrix, $\sum_j d_j^2/(d_j^2 + \lambda)$, is the effective number of parameters — equal to $p$ at $\lambda = 0$ and decreasing smoothly toward $0$. That is the soft version of dropping principal components I was after, and it falls straight out of the L2 penalty. There is a third reading that pins down what $\lambda$ means: put an isotropic Gaussian prior $\beta \sim N(0, (\sigma^2/\lambda)I)$ and the Gaussian likelihood $y\mid\beta \sim N(X\beta, \sigma^2 I)$; the log-posterior is, up to constants, $-\tfrac{1}{2\sigma^2}\|y - X\beta\|^2 - \tfrac{\lambda}{2\sigma^2}\|\beta\|^2$, whose maximizer is the ridge objective, and because everything is Gaussian and conjugate the posterior is Gaussian with mean and mode both equal to $(X'X + \lambda I)^{-1}X'y$. So $\lambda$ is simultaneously the penalty strength, the inverse constraint budget, and the prior precision — the strength of my belief that the coefficients are small — and the spherical $\lambda I$ is the maximally noncommittal prior, equal precision on every coordinate.

Now the elephant: this estimator is biased, and Gauss-Markov just told me the unbiased least-squares estimator is the best linear unbiased one. Writing $W = (X'X + \lambda I)^{-1}X'X$, I have $\hat\beta(\lambda) = W\hat\beta$ and $\mathbb{E}[\hat\beta(\lambda)] = W\beta \ne \beta$ — biased, confirmed. But Gauss-Markov only crowns least squares *within the unbiased class*; by stepping outside it deliberately I am in territory the theorem does not govern, and the real question is which estimator has smaller total error, $\mathrm{MSE} = \text{variance} + \text{bias}^2$. The precedent that this can pay is Stein's: in $p \ge 3$ dimensions the unbiased estimate of a Gaussian mean is inadmissible — biased shrinkage toward a point strictly lowers risk everywhere. Decomposing the ridge MSE,

$$\mathrm{MSE}[\hat\beta(\lambda)] = \sigma^2\, \mathrm{tr}\!\left[W (X'X)^{-1} W'\right] + \beta'(W - I)'(W - I)\beta,$$

the variance term decreases from $\sigma^2\,\mathrm{tr}[(X'X)^{-1}]$ toward $0$ while the squared-bias term rises from $0$ toward $\beta'\beta$. Working instead with the full second-moment matrices $M(\lambda) = \mathbb{E}[(\hat\beta(\lambda) - \beta)(\hat\beta(\lambda) - \beta)']$ and grinding the algebra, factoring $(X'X + \lambda I)^{-1}$ out on both sides, gives

$$M(0) - M(\lambda) = \lambda (X'X + \lambda I)^{-1}\left[\, 2\sigma^2 I + \lambda\sigma^2 (X'X)^{-1} - \lambda\,\beta\beta' \,\right](X'X + \lambda I)^{-1},$$

which is positive definite exactly when the bracket is. The middle term $\lambda\sigma^2(X'X)^{-1}$ is itself positive definite and can only help, so it suffices that $2\sigma^2 I - \lambda\beta\beta' > 0$; since $\beta\beta'$ is rank one with single nonzero eigenvalue $\beta'\beta$, this holds for

$$0 < \lambda < \frac{2\sigma^2}{\beta'\beta}.$$

Throughout that interval $M(0) - M(\lambda)$ is positive definite, so ridge beats OLS in mean squared error under every nonzero positive-semidefinite quadratic weighting; for $\beta = 0$ the rank-one subtraction vanishes and the conclusion is easier. The engine is visible in the formula: the variance saving is first-order in $\lambda$ (the $2\sigma^2 I$ term) while the bias cost is second-order (the $\lambda\beta\beta'$ term), so for small $\lambda$ the saving dominates. In the orthonormal case $X'X = I$ the scalar MSE is $(1+\lambda)^{-2}[p\sigma^2 + \lambda^2\beta'\beta]$, minimized at $\lambda^* = p\sigma^2/(\beta'\beta)$ — strictly positive, larger when there is more noise or more coordinates relative to signal. That scalar optimum and the matrix-dominance interval are different facts answering different questions; both merely confirm that a positive penalty wins.

A few design choices make the budget mean what I want. The penalty $\sum_j \beta_j^2$ is only fair if the coefficients are comparable, so I standardize the columns of $X$ to zero mean and unit variance; otherwise a covariate in tiny units carries a large coefficient and gets clobbered for a reason that has nothing to do with the data. I center $y$ and leave the intercept unpenalized, because shrinking it would make adding a constant to every $y_i$ change the solution, and the location of $y$ is not something I hold a prior-toward-zero belief about. I use the L2 norm rather than L1 for three reasons that I can now see clearly: its linear gradient is what gives the closed form and strict convexity; it is rotationally natural in the eigenbasis and produces the per-direction factor $d_j^2/(d_j^2 + \lambda)$; and it corresponds to the Gaussian prior. The price is that the factor is strictly positive for every finite $\lambda$, so ridge never zeroes a coefficient — it shrinks, it does not select — which is exactly right when I want to keep and balance correlated covariates. As for choosing $\lambda$, the optima depend on the unknown $\beta$ and $\sigma^2$, so I cannot read it off a formula; in practice I trace the coefficient paths $\hat\beta_j(\lambda)$ and take the smallest $\lambda$ past which they stabilize, or I choose $\lambda$ by cross-validation. The existence theorem tells me why a positive penalty is worth searching over; held-out error picks the value that predicts best on the data at hand. The estimator realizes either as the regularized normal-equations solve $(X'X + \lambda I)^{-1}X'y$ — by Cholesky, or by the SVD when stability matters — or as a single linear layer trained with squared-error loss and AdamW weight decay set to $\lambda$ on the slope weights.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- (a) classical closed form ----
class RidgeRegression:
    """Minimize ||y - X w||^2 + lam * ||w||^2; w = (X'X + lam I)^{-1} X'y."""

    def __init__(self, lam=1.0):
        self.lam = lam
        self.w = None

    def fit(self, X, y):
        # Assumes X standardized, y centered (intercept handled outside, unpenalized).
        p = X.shape[1]
        A = X.T @ X + self.lam * np.eye(p)   # PD, invertible even when X'X is singular
        b = X.T @ y
        self.w = np.linalg.solve(A, b)       # solve (X'X + lam I) w = X'y
        return self

    def fit_svd(self, X, y):
        # Numerically stabler: shrink each singular direction by d^2/(d^2+lam).
        U, d, Vt = np.linalg.svd(X, full_matrices=False)
        self.w = (Vt.T * (d / (d**2 + self.lam))) @ (U.T @ y)
        return self

    def predict(self, X):
        return X @ self.w


# ---- (b) linear head trained with AdamW weight decay ----
class RidgeHead(nn.Module):
    """A single linear layer trained with MSE loss and weight decay on slopes."""

    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, features):
        return self.linear(features).squeeze(-1)   # [B]


def train_ridge_head(model, loader, lam):
    # Use AdamW weight_decay as the ridge penalty on slopes only; leave bias unpenalized.
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.linear.weight], "weight_decay": lam},
            {"params": [model.linear.bias], "weight_decay": 0.0},
        ],
        lr=1e-3,
    )
    model.train()
    for features, target in loader:
        optimizer.zero_grad()
        loss = F.mse_loss(model(features), target)   # data term
        loss.backward()                              # AdamW applies decoupled shrinkage to slope weights
        optimizer.step()
    return model
```
