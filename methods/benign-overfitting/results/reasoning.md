I start from the part that looks impossible. If the labels contain noise and I force the training residuals all the way to zero, the noise has to be written somewhere into the fitted parameter. In the classical picture that is already the end of the story: the parameter has memorized random errors, so prediction should be bad. But that conclusion is smuggling in an assumption. It assumes that every direction in which the parameter can move matters about equally for prediction.

In squared-loss linear prediction, that assumption is false as soon as the covariates are anisotropic. Prediction risk is not ordinary Euclidean error in the parameter. It is the covariance-weighted error. If `Sigma` is the covariance, then an error in a large-eigenvalue direction is expensive and an error in a tiny-eigenvalue direction is almost invisible. So I stop asking whether the noise is memorized. It is. I ask where the memorized noise lands, and how much the covariance charges for it.

I take the simplest exact-fit rule that has a chance of being canonical. When there are more directions than samples, the equations `X theta = y` have many solutions, so I choose the one with the smallest norm:

```tex
\hat\theta = X^\top(XX^\top)^{-1}y.
```

This rule is the zero-regularization limit of ridge regression in the full-row-rank regime. It also gives me algebra I can trust. With `y = X theta^* + epsilon`, I can write

```tex
\hat\theta
  = X^\top(XX^\top)^{-1}X\theta^*
    + X^\top(XX^\top)^{-1}\epsilon.
```

Let

```tex
P = X^\top(XX^\top)^{-1}X.
```

Then `P` is the projection onto the row space seen by the sample, and

```tex
\theta^*-\hat\theta
  = (I-P)\theta^* - X^\top(XX^\top)^{-1}\epsilon.
```

Now the two errors are separated. The first term is signal the sample does not see. The second term is pure training noise pushed back into parameter space by the pseudoinverse.

The excess risk of any fitted parameter is

```tex
R(\hat\theta)
  = E_x (x^\top(\theta^*-\hat\theta))^2
  = (\theta^*-\hat\theta)^\top\Sigma(\theta^*-\hat\theta).
```

So the split becomes two covariance-weighted quantities:

```tex
B = (I-P)\Sigma(I-P),
\qquad
C = (XX^\top)^{-1}X\Sigma X^\top(XX^\top)^{-1}.
```

The signal term is `theta^{*T} B theta^*`. The noise term is `epsilon^T C epsilon`, and in expectation it is at least `sigma^2 tr(C)`. This is the first real simplification: the whole question of whether exact fitting of noise is harmful is now mostly the question of whether `tr(C)` is small.

The bias term is not the mystery. Because `(I-P)X^T = 0`, I can subtract the empirical covariance inside the quadratic form:

```tex
(I-P)\Sigma(I-P)
  = (I-P)(\Sigma - n^{-1}X^\top X)(I-P)
```

inside the relevant expression. Projection has norm at most one, so this term is bounded by a sample-covariance deviation times `||theta^*||^2`. The usual effective scale `tr(Sigma)/||Sigma||` has to be small compared with `n`, up to concentration factors. That is the condition saying the signal is not too large for the sample to see.

The variance term is where the strange part lives. I diagonalize the covariance. Write `Sigma v_i = lambda_i v_i`, and define

```tex
z_i = Xv_i/\sqrt{\lambda_i}.
```

Under the model assumptions, the `z_i` are independent standardized subgaussian vectors in `R^n`. Then

```tex
A = XX^\top = \sum_i \lambda_i z_i z_i^\top,
\qquad
X\Sigma X^\top = \sum_i \lambda_i^2 z_i z_i^\top,
```

and therefore

```tex
tr(C) = \sum_i \lambda_i^2 z_i^\top A^{-2}z_i.
```

This expression says exactly how each covariance direction contributes to the cost of memorizing label noise. But `z_i` also appears inside `A`, so I need to decouple one direction at a time. Splitting

```tex
A = \lambda_i z_i z_i^\top + A_{-i}
```

and using Sherman-Morrison gives

```tex
\lambda_i^2 z_i^\top A^{-2}z_i
=
\frac{\lambda_i^2 z_i^\top A_{-i}^{-2}z_i}
{(1+\lambda_i z_i^\top A_{-i}^{-1}z_i)^2}.
```

Now `z_i` is independent of `A_{-i}`, so quadratic forms can concentrate. This is the place where the geometry has to enter.

I check the failure case first. If `p = n` and `Sigma = I`, then there are exactly enough equally important directions to interpolate. The noise cannot be moved into a cheap subspace. In that case `C = (XX^T)^{-1}`, and for a square Gaussian design the trace is bounded below by a constant, often worse near exact squareness. Exact fitting is harmful. This tells me the phenomenon cannot be explained by interpolation alone, nor by overparameterization counted naively. I need extra directions whose covariance weights are small.

So I split the spectrum. The first `k` directions are the directions I may need for signal. The tail after `k` is the possible reservoir for noise. Define

```tex
A_k = \sum_{i>k}\lambda_i z_i z_i^\top.
```

Since each `z_i z_i^T` has expectation `I`, the expected tail matrix is

```tex
(\sum_{i>k}\lambda_i) I.
```

But concentration has an error governed by the largest tail weight, `lambda_{k+1}`, times the sample dimension `n`. So the tail behaves like a scalar matrix only when the total tail mass dominates that largest tail weight times `n`:

```tex
\sum_{i>k}\lambda_i >> n\lambda_{k+1}.
```

This is the first effective-rank condition:

```tex
r_k(\Sigma) = \frac{\sum_{i>k}\lambda_i}{\lambda_{k+1}}.
```

When `k` is below a constant fraction of `n` and `r_k(Sigma) >= b n`, the tail Gram matrix has all its eigenvalues within constant factors of its mean scale. That is the formal meaning of "many low-variance directions." The tail is not merely small; it is wide enough to look isotropic in the `n`-dimensional sample space.

I let `k^*` be the first index where this happens:

```tex
k^* = \min\{k : r_k(\Sigma) >= b n\}.
```

If there is no such index, I read `k^*` as infinity. If `k^*` is infinite or at least a constant fraction of `n`, then too many costly directions appear before the reservoir starts. The noise has no early cheap place to spread, and the trace stays bounded below by a constant. That recovers the harmful interpolation threshold.

If `k^*` is small, I can bound the trace by splitting again at some `l <= k^*`. The leading `l` terms each cost on the order of `1/n`, because the denominator in the Sherman-Morrison expression is stabilized by the isotropic tail. They contribute `l/n`. The remaining tail terms cost

```tex
n \frac{\sum_{i>l}\lambda_i^2}
{(\lambda_{k^*+1} r_{k^*}(\Sigma))^2}.
```

The denominator is the squared tail mass, because

```tex
\lambda_{k^*+1} r_{k^*}(\Sigma)
= \sum_{i>k^*}\lambda_i.
```

Optimizing over `l` puts the split at `k^*`. The tail term becomes

```tex
n \frac{\sum_{i>k^*}\lambda_i^2}
{(\sum_{i>k^*}\lambda_i)^2}.
```

That ratio needs a second effective rank:

```tex
R_k(\Sigma)
  =
\frac{(\sum_{i>k}\lambda_i)^2}{\sum_{i>k}\lambda_i^2}.
```

Now the variance cost is

```tex
tr(C) \asymp k^*/n + n/R_{k^*}(\Sigma),
```

up to constants, with a matching lower bound. The first term is the price of the important directions before the reservoir. The second term is the price of how balanced the reservoir is. If the tail consists of many comparable tiny eigenvalues, `R_{k^*}` is large and the noise is diluted. If one tail eigenvalue dominates, the noise concentrates and the cost remains visible.

This finally explains why one effective rank is not enough. `r_k` answers whether the low-variance reservoir starts early enough and is wide enough to stabilize the sample Gram matrix. `R_k` answers whether the reservoir is balanced enough to dilute noise. A large raw dimension does not guarantee either property.

The sufficient benign condition now has three parts. The signal scale must be controlled:

```tex
r_0(\Sigma_n)/n -> 0.
```

The reservoir must start before a linear fraction of the sample size:

```tex
k_n^*/n -> 0.
```

And the reservoir must be balanced enough:

```tex
n/R_{k_n^*}(\Sigma_n) -> 0.
```

I also have to keep the converse straight. If `r_0(Sigma)/(n log(1+r_0(Sigma)))` is large, some signal vector of the same norm must have visible excess risk. So I should not pretend the signal side is an exact if-and-only-if at the finite-sample level. The sharp matching story is clearest for the two tail quantities controlling noise absorption.

This is more than observing double descent. Double descent says risk may go down after interpolation. This explains when and why it goes down: the minimum-norm bias separates signal from noise, the covariance metric makes some directions prediction-cheap, and a high-dimensional balanced tail lets the interpolant hide noise where future prediction barely sees it.

The eigenvalue examples make the geometry concrete. In a fixed infinite-dimensional space, the eigenvalues have to decay almost exactly at the boundary of summability. If they decay too slowly, the signal scale is too large. If they decay too quickly, the low-variance tail is too thin to dilute noise. A sequence like `lambda_k = k^{-1} log^{-\beta}(k+1)` with `beta > 1` threads that needle.

In a large finite-dimensional space, the story is less fragile. A small isotropic floor over many dimensions creates a broad flat reservoir. Then even a rapidly decaying signal spectrum can coexist with harmless interpolation, provided the dimension grows faster than the sample size and the total floor mass is small but not exponentially tiny. That is the scientific lesson: finite but very high dimension is not a minor technical variant of infinite dimension; it changes how easy it is for exact interpolation to be harmless.
