Let me start from what actually goes wrong, because the model itself is the most innocent thing imaginable: `y = X beta + eps`, a linear predictor, noise with mean zero and covariance `sigma^2 I`. I want `beta`. The recipe is two hundred years old — minimize the residual sum of squares `||y - X beta||^2`, set the gradient to zero, get the normal equations `X'X beta = X'y`, and if `X'X` inverts, `beta_hat = (X'X)^{-1} X'y`. It's unbiased, it's the maximum-likelihood estimate under Gaussian noise, and Gauss-Markov tells me it's the best linear unbiased estimator there is. By that standard I should be done. And on a clean, orthogonal design I am done. The trouble is the data I actually have: the columns of `X` are correlated, nearly redundant, sometimes there are more of them than I have rows. And there least squares falls apart in a way the optimality theorem completely fails to warn me about.

Let me see exactly *how* it falls apart, because the shape of the failure is going to tell me the shape of the cure. The estimator's quality lives in its covariance, `Cov(beta_hat) = sigma^2 (X'X)^{-1}`. So I should look hard at `(X'X)^{-1}`. `X'X` is symmetric positive semidefinite, so diagonalize it: `X'X = sum_j d_j^2 v_j v_j'`, where the `v_j` are orthonormal eigenvectors and `d_j^2` the eigenvalues — and the `d_j` are just the singular values of `X` itself, since if `X = U D V'` then `X'X = V D'D V' = sum_j d_j^2 v_j v_j'`. Inverting a symmetric matrix inverts its eigenvalues on the same eigenvectors, so

```
Cov(beta_hat) = sigma^2 (X'X)^{-1} = sigma^2 sum_j d_j^{-2} v_j v_j'.
```

There it is in black and white. The variance of the estimate along direction `v_j` is `sigma^2 / d_j^2`. A direction with a *small* singular value `d_j` gets variance blown up by `1/d_j^2`. And a small singular value is *exactly* what collinearity is: if two covariates are nearly proportional, there's a combination of them, some `v_j`, along which `X` barely moves — `X v_j` is tiny — so `d_j` is tiny, so the data carries almost no information about `beta` in that direction, and the noise fills the vacuum. That's why my coefficients come out with insane magnitudes and signs that flip if I jiggle one data point: the estimate along the ill-conditioned direction is essentially `sigma^2 / d_j^2` worth of pure noise. Push it to the limit — exact dependence, or `p > n` — and some `d_j` is literally zero, `(X'X)^{-1}` doesn't exist, the normal equations have an entire affine space of solutions, and least squares doesn't even return an answer. So the disease has a precise diagnosis: tiny (or zero) singular values of `X`, i.e. tiny (or zero) eigenvalues of `X'X`, and the `1/d_j^2` amplification they cause.

Now, my first instinct is "just don't invert a near-singular matrix." Could I drop the offending directions? Throw away the covariates that are collinear, or — cleaner — project onto the top principal components of `X`, the high-`d_j` directions, and regress only there. That does kill the `1/d_j^2` blow-up, because I've deleted the small-`d_j` directions outright. But it's a hard cut, and two things bother me about it. First, it's all-or-nothing: a direction is kept whole or discarded whole, so my estimator is a discontinuous, high-variance function of where I draw the threshold, and a small data change can flip which direction crosses it. Second, and worse, I'm deleting directions by their variance *in `X`*, with no reference to whether they help predict `y`. A direction can have a small `d_j` and still be genuinely predictive; principal-component regression throws it out with the noise. I don't want a guillotine that severs directions; I want something that *damps* the unstable ones smoothly, in proportion to how unstable they are, while barely touching the stable ones. A dial, not a switch.

So what's the smallest, most honest thing I can do to `(X'X)^{-1}` to stop the `1/d_j^2` explosion? The explosion is `d_j^{-2}` for small `d_j`. If I could replace `d_j^2` by `d_j^2 + (something positive)` before inverting, then for the healthy big-`d_j` directions the "something" is negligible and nothing changes, but for the sick small-`d_j` directions the denominator can't fall below that floor, so the variance is capped instead of diverging. And `d_j^2 + lambda` on every eigenvalue is, in matrix terms, just adding `lambda` to the whole spectrum — i.e. replacing `X'X` by `X'X + lambda I`. That single move does several things at once. `X'X` is positive semidefinite, `lambda I` for `lambda > 0` is positive definite, and a PSD matrix plus a PD matrix is positive definite — so `X'X + lambda I` has all eigenvalues at least `lambda`, no zero eigenvalue, and is *always invertible*, even when `X'X` was singular, even when `p > n`. The estimator

```
beta_hat(lambda) = (X'X + lambda I)^{-1} X' y
```

is therefore well-defined across the entire range from orthogonal to super-collinear designs, and along each eigendirection it inverts `d_j^2 + lambda` instead of `d_j^2`, which is the smooth, proportional damping I wanted. At this point it looks like an ad-hoc patch — "nudge the diagonal up until the matrix inverts." I don't trust patches, so I need to know what objective, if any, this is the honest optimum of.

Let me reverse-engineer it. Suppose I'm not minimizing the plain residual sum of squares but a *penalized* version that also charges me for large coefficients: `L(beta) = ||y - X beta||^2 + lambda ||beta||^2`. Differentiate. The first term's gradient is `-2 X'(y - X beta)`; the second term's gradient is `2 lambda beta`. So

```
dL/dbeta = -2 X'(y - X beta) + 2 lambda beta = -2 X'y + 2 (X'X + lambda I) beta.
```

Set it to zero: `(X'X + lambda I) beta = X'y`, hence `beta = (X'X + lambda I)^{-1} X'y`. So my diagonal patch is *exactly* the minimizer of least-squares-plus-an-L2-penalty-on-the-coefficients. It wasn't a hack on the linear algebra at all; it was the closed-form solution of a perfectly sensible objective: fit the data, but pay a price proportional to `lambda` for the squared length of your coefficient vector. And it's a genuine global minimum, not just a stationary point: the Hessian is `d^2 L / dbeta dbeta' = 2 (X'X + lambda I)`, positive definite for `lambda > 0`, so `L` is strictly convex and the solution is the unique global minimizer. That reframes the parameter `lambda` immediately — it's not a fudge factor on a matrix, it's the *strength of the penalty*, the exchange rate between fitting and shrinking. At `lambda = 0` I recover least squares; as `lambda -> infinity` the penalty dominates and `beta_hat(lambda) -> 0`.

There's a second reading of the same objective that gives me the geometry. Minimizing `||y - X beta||^2 + lambda ||beta||^2` is, by Lagrangian duality, the same as minimizing `||y - X beta||^2` subject to a hard budget `||beta||^2 <= c`, with `lambda` the Karush-Kuhn-Tucker multiplier on the constraint — bigger `lambda` corresponds to a tighter budget `c`. Picture it: the residual-sum-of-squares contours are ellipsoids centered at the unconstrained least-squares solution `beta_hat`, and I'm forced to stay inside a ball around the origin. The penalized solution is where the smallest ellipsoid touches the ball — a *shrunk* version of `beta_hat`, pulled toward zero. So all three pictures agree: a diagonal-loaded inverse, an L2-penalized least-squares fit, and a norm-constrained fit are the same estimator, shrinking the least-squares answer toward the origin by an amount governed by `lambda`.

I want to see the shrinkage explicitly, direction by direction, so let me push the singular value decomposition `X = U D V'` through both estimators. For least squares, `beta_hat = (X'X)^{-1} X' y = (V D'D V')^{-1} V D' U' y = V (D'D)^{-1} D' U' y`, which along eigendirection `j` reads off the coordinate `u_j' y / d_j`. For the penalized version,

```
beta_hat(lambda) = (X'X + lambda I)^{-1} X' y = (V D'D V' + lambda V V')^{-1} V D' U' y
                  = V (D'D + lambda I)^{-1} D' U' y,
```

because `V V' = I` lets me pull the `V`'s out and combine the brackets. Coordinate by coordinate, the contribution that was `d_j^{-1} (u_j' y)` becomes `d_j (d_j^2 + lambda)^{-1} (u_j' y)`. So relative to least squares, each direction is multiplied by

```
d_j^2 / (d_j^2 + lambda),
```

a factor in `(0, 1]`. For a big `d_j` (well-conditioned direction) this is `d_j^2 / (d_j^2 + lambda) ~ 1` — barely shrunk. For a tiny `d_j` (the directions that were exploding) it's `~ d_j^2 / lambda ~ 0` — shrunk almost to nothing. The estimator damps each direction in exact proportion to how unstable it is. The corresponding fit is `y_hat(lambda) = X beta_hat(lambda) = sum_j u_j [d_j^2 / (d_j^2 + lambda)] (u_j' y)`, the least-squares projection with each principal-direction coordinate scaled down by that same factor. And there's a clean summary number hiding here: the trace of the fit's hat matrix, `sum_j d_j^2 / (d_j^2 + lambda)`, counts the *effective* number of parameters — it equals `p` at `lambda = 0` (full least squares) and decreases smoothly toward 0 as `lambda` grows. So instead of the guillotine of principal-component regression, which sets the factor to exactly 1 for kept directions and exactly 0 for dropped ones, I have a soft version that interpolates continuously and lets each direction's data speak in proportion to its conditioning. That's the dial I wanted, and it falls straight out of the L2 penalty.

But now I have to confront the elephant, because what I've built is *biased*, and Gauss-Markov just told me the unbiased least-squares estimator is the best linear unbiased one. Let me write the bias out so I know what I'm dealing with. Define the linear operator `W = (X'X + lambda I)^{-1} X'X`. Then `beta_hat(lambda) = (X'X + lambda I)^{-1} X'y = (X'X + lambda I)^{-1} X'X (X'X)^{-1} X'y = W beta_hat`, so the penalized estimator is just `W` applied to the least-squares estimator. Taking expectations, `E[beta_hat(lambda)] = W beta`, which is not `beta` for any `lambda > 0` — biased, confirmed. So am I dead on arrival? No — and this is the crucial point — Gauss-Markov only crowns least squares within the class of *unbiased* linear estimators. By stepping outside that class deliberately, I'm not refuted by the theorem; I'm in territory it doesn't govern. The real question isn't "biased or unbiased," it's "which estimator has smaller total error?" — and total error is mean squared error, `MSE = variance + bias^2`. Least squares is the zero-bias, all-variance corner. I'm betting I can buy a large reduction in variance for a small payment in bias and come out ahead in the sum. I have a precedent that makes me believe this is possible and not wishful: in several dimensions, the plain unbiased estimate of a Gaussian mean is *inadmissible* — shrinking it toward a point strictly lowers total squared-error risk everywhere, precisely because the shrinkage estimator is biased and so escapes the unbiased-optimality trap. Same logic, same escape hatch. So I need to actually compute the MSE of `beta_hat(lambda)` and prove a beneficial trade exists.

Let me derive the MSE. I'll measure it as `E[(beta_hat(lambda) - beta)'(beta_hat(lambda) - beta)]`, the expected squared distance to the truth. Write `beta_hat(lambda) = W beta_hat` and split into the variance about its own mean plus the squared bias. The standard decomposition for `E[(W beta_hat - beta)'(W beta_hat - beta)]` is: add and subtract the mean `W beta`, the cross term vanishes in expectation, and I'm left with `E[(W beta_hat - W beta)'(W beta_hat - W beta)] + (W beta - beta)'(W beta - beta)`. The first piece is the variance contribution; using `beta_hat ~ (beta, sigma^2 (X'X)^{-1})` and the identity `E[z' A z] = tr(A Cov(z)) + (E z)' A (E z)` for the centered part, it becomes `sigma^2 tr[W (X'X)^{-1} W']`. The second piece is `beta'(W - I)'(W - I)beta`, the squared bias. So

```
MSE[beta_hat(lambda)] = sigma^2 tr[W (X'X)^{-1} W']  +  beta'(W - I)'(W - I)beta,
                          \_______ variance _______/     \________ bias^2 ________/
```

with `W = (X'X + lambda I)^{-1} X'X`. Sanity check the two ends. At `lambda = 0`, `W = I`, so the bias term is zero and the variance term is `sigma^2 tr[(X'X)^{-1}]` — that's exactly OLS, all variance, no bias, good. As `lambda -> infinity`, `W -> 0`, the variance term vanishes and the bias term goes to `beta'beta` — that's the MSE of the estimator that just guesses zero, all bias, no variance, also good. So the variance term starts positive and decreases toward 0 with `lambda`, the squared-bias term starts at 0 and increases toward `beta'beta`, and the MSE is their sum. The whole question of whether ridging can win is the question of which term moves faster near `lambda = 0`.

Let me get a clean, fully explicit instance first to build intuition, then prove it in general. Take `X` orthonormal, so `X'X = I`. Then `W = (I + lambda I)^{-1} I = (1 + lambda)^{-1} I`, the variance term is `sigma^2 tr[(1+lambda)^{-2} I] = p sigma^2 (1+lambda)^{-2}`, and the bias term is `beta'((1+lambda)^{-1} - 1)^2 beta = (lambda/(1+lambda))^2 beta'beta = lambda^2 (1+lambda)^{-2} beta'beta`. So

```
MSE[beta_hat(lambda)] = (1 + lambda)^{-2} [ p sigma^2 + lambda^2 beta'beta ].
```

Differentiate with respect to `lambda` and set to zero. The numerator-of-the-derivative works out to `-2(1+lambda)^{-3}[p sigma^2 + lambda^2 beta'beta] + (1+lambda)^{-2} 2 lambda beta'beta`; multiplying through by `(1+lambda)^3 / 2` and setting to zero gives `-(p sigma^2 + lambda^2 beta'beta) + lambda(1+lambda)beta'beta = 0`, i.e. `-p sigma^2 - lambda^2 beta'beta + lambda beta'beta + lambda^2 beta'beta = 0`, so `lambda beta'beta = p sigma^2`, and the optimum is

```
lambda* = p sigma^2 / (beta'beta).
```

It's strictly positive whenever `sigma^2 > 0` and `beta'beta > 0`. So in the orthonormal case the best scalar penalty is *never* `lambda = 0` — it is larger when the noise `sigma^2` or the number of coordinates `p` is larger, and smaller when the true signal length `beta'beta` is larger. That's deeply encouraging: it says the all-variance corner is provably suboptimal, and it gives the optimal trade an interpretation — penalize more when there is more noise relative to signal. But it's also a tease, because `lambda*` depends on the unknown `beta` and `sigma^2`; I'll have to come back to how to choose `lambda` without knowing them.

Now the general theorem, because I want to know this isn't an orthonormal fluke. I should state the comparison only where the ordinary least-squares estimator itself exists, so take `X'X` positive definite. The cleanest route is through the second-moment matrices rather than the scalar MSE — define `M(lambda) = E[(beta_hat(lambda) - beta)(beta_hat(lambda) - beta)']`, the full `p x p` mean-squared-error matrix (variance plus bias-outer-product), and show `M(0) - M(lambda)` is positive definite for small `lambda`; then every positive-semidefinite quadratic loss weighting inherits the inequality. So compute the difference. `M(0) = sigma^2 (X'X)^{-1}` (pure OLS variance). `M(lambda) = sigma^2 W (X'X)^{-1} W' + (W - I) beta beta' (W - I)'`. Subtract, and grind through the algebra — the key is to factor out `(X'X + lambda I)^{-1}` on the left and its transpose on the right. After expanding `W = (X'X + lambda I)^{-1} X'X` and collecting terms, the variance difference `sigma^2[(X'X)^{-1} - W(X'X)^{-1}W']` becomes `sigma^2 (X'X+lambda I)^{-1}[2 lambda I + lambda^2 (X'X)^{-1}](X'X+lambda I)^{-1}`, and the squared-bias term being subtracted is `lambda^2 (X'X+lambda I)^{-1} beta beta' (X'X+lambda I)^{-1}`. Pulling out the common factors,

```
M(0) - M(lambda) = lambda (X'X + lambda I)^{-1} [ 2 sigma^2 I + lambda sigma^2 (X'X)^{-1} - lambda beta beta' ] (X'X + lambda I)^{-1}.
```

The outer factors `(X'X+lambda I)^{-1}` are positive definite, so the whole thing is positive definite if and only if the bracket `2 sigma^2 I + lambda sigma^2 (X'X)^{-1} - lambda beta beta'` is positive definite. The middle term `lambda sigma^2 (X'X)^{-1}` is itself positive definite, so it can only help; it suffices to show `2 sigma^2 I - lambda beta beta' > 0`. Now `beta beta'` is a rank-one matrix with single nonzero eigenvalue `beta'beta` (eigenvector `beta`); subtracting `lambda beta beta'` from `2 sigma^2 I` keeps everything positive exactly when that one eigenvalue isn't overpowered, i.e. when `lambda beta'beta < 2 sigma^2`, that is

```
0 < lambda < 2 sigma^2 / (beta'beta).
```

For every `lambda` in that interval, `M(0) - M(lambda)` is positive definite, hence ridge beats OLS in mean squared error under any nonzero positive-semidefinite quadratic weighting. If `beta = 0`, the rank-one subtraction disappears and the same small-positive-`lambda` conclusion is even easier. So the existence result is exactly the thing I needed: whenever the ordinary least-squares comparison is meaningful and there is noise, there is a positive penalty that does strictly better than least squares. I should not confuse this matrix-dominance interval with the scalar optimum from the orthonormal calculation. The orthonormal scalar optimum is `p sigma^2/(beta'beta)`, while this interval is the conservative range that makes the full matrix difference positive definite; both facts point in the same direction near zero, but they answer different questions. This is the payoff: I haven't violated Gauss-Markov, I've gone around it, and the trip is provably worth taking. The variance reduction near `lambda = 0` is first-order in `lambda` (the `2 lambda I` term) while the bias cost is second-order (the `lambda^2 beta beta'` term), so for small `lambda` the variance saving dominates — that's the engine of the theorem.

There's a third interpretation I should pin down, because it tells me what `lambda` *means* and how to think about choosing it: the Bayesian one. Put a prior on `beta` that says, before seeing data, I expect it to be near zero — specifically `beta ~ N(0, (sigma^2/lambda) I)`, an isotropic Gaussian centered at the origin with precision proportional to `lambda`. Combine it with the Gaussian likelihood `y | beta ~ N(X beta, sigma^2 I)`. The log-posterior is, up to constants, `-(1/2sigma^2)||y - X beta||^2 - (lambda/2 sigma^2)||beta||^2`, and maximizing it (the MAP estimate) is identical to minimizing `||y - X beta||^2 + lambda ||beta||^2` — the ridge objective exactly. Moreover, because everything is Gaussian and conjugate, the full posterior is Gaussian and its mean is `(X'X + lambda I)^{-1}X'y = beta_hat(lambda)`, so the ridge estimate is simultaneously the posterior mode and the posterior mean. This is the most satisfying reading: `lambda` is the *prior precision*, the strength of my prior belief that the coefficients are small. A large `lambda` is a confident "they're near zero" prior; `lambda = 0` is a flat, infinitely-uninformative prior that reduces to least squares. The penalty isn't an arbitrary regularizer — it's a quantified prior, and the bias it introduces is just the prior pulling the estimate toward its mean of zero. It also explains *why isotropic* `lambda I` and not something fancier: with no reason to favor one coefficient direction over another a priori (especially after I standardize the columns), the maximally noncommittal prior is spherical, equal precision on every coordinate, which is exactly `lambda I`.

That standardization point deserves a beat, because the penalty `||beta||^2 = sum_j beta_j^2` is only fair if the coefficients are comparable. If covariate `j` is measured in tiny units, its coefficient is large to compensate, and the penalty would clobber it for a reason that has nothing to do with the data — a unit change, not a signal change. So I standardize the columns of `X` to zero mean and unit sample variance before fitting, which puts the coefficients on a common scale and makes `sum_j beta_j^2` a meaningful single budget. I also center `y` and leave the intercept *unpenalized*: if I shrank the intercept, then adding a constant to every `y_i` would change the solution in a way it shouldn't — the location of `y` is not something I have a prior-toward-zero belief about. So: standardize features, center the response, penalize the slopes but not the intercept. These aren't decoration; each one is forced by wanting the penalty to mean what I think it means.

Why the L2 norm specifically, and not, say, the sum of absolute values? Three reasons, all of which I can now see clearly. First, `||beta||^2` is what gives me the closed form and the strict convexity — its gradient is linear in `beta`, so the stationarity equation stays linear and inverts to `(X'X + lambda I)^{-1}X'y`. An absolute-value penalty has a kink at zero, no closed form, and a qualitatively different solution. Second, the L2 penalty is rotationally natural in the eigenbasis: it's what produces the clean per-direction shrinkage factor `d_j^2/(d_j^2+lambda)` that damps each principal direction in proportion to its conditioning, which is precisely the behavior I diagnosed as needed. Third, it corresponds to the Gaussian prior, the maximally noncommittal "small and smooth" belief. The cost is that it never sets a coefficient exactly to zero — the shrinkage factor `d_j^2/(d_j^2+lambda)` is strictly positive for every finite `lambda`, so ridge keeps all variables, just smaller; it does prediction-by-shrinkage, not variable selection. For my purpose — stabilize an ill-conditioned fit and predict well — that's exactly the right tradeoff; I *want* to keep and balance correlated covariates rather than arbitrarily delete one.

The one loose thread is choosing `lambda`. The theory hands me the optimum `lambda* = p sigma^2/(beta'beta)` in the orthonormal scalar-MSE case and the matrix-improvement interval `(0, 2 sigma^2/beta'beta)` in the full-rank comparison, but both depend on the unknowns `beta` and `sigma^2`. So I can't read `lambda` off a formula. Two practical answers. A diagnostic one: trace the coefficient paths `beta_hat_j(lambda)` as `lambda` sweeps from 0 upward — at `lambda = 0` they're the wild least-squares values, and as `lambda` grows they march toward zero, with the unstable ones snapping into sensible ranges first; pick the smallest `lambda` past which the paths have settled into stability. The data-driven one I'd actually automate: choose `lambda` by cross-validation, i.e. pick the value that minimizes held-out prediction error. The theorem tells me why a positive penalty is worth searching over; the held-out criterion decides which positive penalty predicts best in the data at hand.

Now let me land this on code I'd actually run. There are two matching ways to compute or train toward `beta_hat(lambda) = (X'X + lambda I)^{-1}X'y`. The direct way is to form and solve the regularized normal equations — that's the classical closed form, and it's what a standard linear-model library does (it solves `(X'X + lambda I) w = X'y` by a Cholesky factorization, or via the SVD `w = V diag(d_j/(d_j^2+lambda)) U' y` when stability matters). The training-harness way is a single linear layer with squared-error loss and shrinkage on the slope parameters. In plain gradient descent, adding `lambda ||w||^2` to the loss gives the exact penalized objective; in an AdamW training loop, passing `weight_decay=lambda` applies the code-level ridge penalty as decoupled shrinkage directly to the weights at each step. So the same pressure appears in two forms: the exact closed-form regularized solve, and an `nn.Linear` head trained with MSE loss plus AdamW weight decay on the weights.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- (a) classical closed form: regularized normal equations ----
class RidgeRegression:
    """Ridge regression: minimize ||y - X w||^2 + lam * ||w||^2.
    Closed form w = (X'X + lam I)^{-1} X'y; always defined for lam > 0."""

    def __init__(self, lam=1.0):
        self.lam = lam            # the L2 penalty strength (Tikhonov / ridge parameter)
        self.w = None

    def fit(self, X, y):
        # Standardize is assumed done upstream; intercept handled outside (center y).
        p = X.shape[1]
        A = X.T @ X + self.lam * np.eye(p)   # X'X + lam I  -> PD, invertible even if X'X singular
        b = X.T @ y                          # X'y
        self.w = np.linalg.solve(A, b)       # solve (X'X + lam I) w = X'y  (Cholesky under the hood)
        return self

    def fit_svd(self, X, y):
        # Equivalent, numerically gentler: shrink each singular direction by d^2/(d^2+lam).
        U, d, Vt = np.linalg.svd(X, full_matrices=False)
        shrink = d / (d**2 + self.lam)       # = (d^2/(d^2+lam)) / d, the per-direction factor / d_j
        self.w = (Vt.T * shrink) @ (U.T @ y)
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
        pred = model(features)
        loss = F.mse_loss(pred, target)        # squared-error data term
        loss.backward()                        # AdamW applies decoupled shrinkage to slope weights
        optimizer.step()
    return model
```

So the causal chain, start to finish: least squares is unbeatable among unbiased linear estimators by Gauss-Markov, but its covariance `sigma^2 sum_j d_j^{-2} v_j v_j'` explodes along the small-singular-value directions that collinearity creates, and is undefined when a singular value hits zero. Diagnosing that the disease is small eigenvalues of `X'X` points straight at the cure: add `lambda I` to the spectrum, which floors every eigenvalue at `lambda`, makes the matrix always invertible, and inverts `d_j^2 + lambda` instead of `d_j^2` — damping each direction by `d_j^2/(d_j^2+lambda)` in proportion to its instability. That diagonal patch turns out to be the exact, strictly-convex global minimizer of the L2-penalized least-squares objective `||y - X beta||^2 + lambda||beta||^2`, equivalently a norm-budget-constrained fit, equivalently the posterior mean/mode under an isotropic Gaussian prior `N(0, (sigma^2/lambda)I)` — so `lambda` is simultaneously the penalty strength, the inverse constraint budget, and the prior precision. The estimator is biased, but Gauss-Markov doesn't apply to biased estimators, and computing `MSE = sigma^2 tr[W(X'X)^{-1}W'] + beta'(W-I)'(W-I)beta` shows the variance term falls first-order in `lambda` while the bias term rises only second-order, so for full-rank OLS comparisons the matrix difference `M(0) - M(lambda) = lambda(X'X+lambda I)^{-1}[2 sigma^2 I + lambda sigma^2(X'X)^{-1} - lambda beta beta'](X'X+lambda I)^{-1}` is positive definite throughout the guaranteed interval `0 < lambda < 2 sigma^2/(beta'beta)` when `beta` is nonzero, with the zero-signal case easier. There is always a positive penalty that strictly beats least squares, and in the orthonormal scalar-MSE case the optimum is `lambda* = p sigma^2/(beta'beta)`. Standardizing features, centering the response, and leaving the intercept unpenalized make the `sum_j beta_j^2` budget meaningful; the L2 norm (not L1) is what buys the closed form, the eigenbasis-proportional shrinkage, and the Gaussian-prior reading. And the whole thing realizes either as the regularized normal-equations solve `(X'X + lambda I)^{-1}X'y` or as a single linear layer trained with MSE loss and AdamW `weight_decay=lambda` on the slope weights.
