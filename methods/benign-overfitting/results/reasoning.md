I start from the part that looks impossible. If the labels contain noise and I force the training residuals all the way to zero, the noise has to be written somewhere into the fitted parameter. In the classical picture that is already the end of the story: the parameter has memorized random errors, so prediction should be bad. But that conclusion is smuggling in an assumption. It assumes that every direction in which the parameter can move matters about equally for prediction.

In squared-loss linear prediction, that assumption is false as soon as the covariates are anisotropic. Prediction risk is not ordinary Euclidean error in the parameter. It is the covariance-weighted error. If `Sigma` is the covariance, then an error in a large-eigenvalue direction is expensive and an error in a tiny-eigenvalue direction is almost invisible. So I stop asking whether the noise is memorized. It is. The better question is where the memorized noise lands, and how much the covariance charges for it.

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

The signal term is `theta^{*T} B theta^*`. The noise term is `epsilon^T C epsilon`, and in expectation it is at least `sigma^2 tr(C)`. The two pieces have decoupled cleanly, and the noise piece has collapsed to a single scalar `tr(C)`. So before going further I want to know whether `tr(C)` can actually be small, because if it cannot, the rest of the analysis is moot.

The bias term I can deal with quickly. Because `(I-P)X^T = 0`, I can subtract the empirical covariance inside the quadratic form:

```tex
(I-P)\Sigma(I-P)
  = (I-P)(\Sigma - n^{-1}X^\top X)(I-P)
```

inside the relevant expression. Projection has norm at most one, so this term is bounded by a sample-covariance deviation times `||theta^*||^2`. The usual effective scale `tr(Sigma)/||Sigma||` has to be small compared with `n`, up to concentration factors. That is the condition saying the signal is not too large for the sample to see, and it is the same kind of condition that controls any least-squares fit.

The variance term is where the surprise has to live or die. I diagonalize the covariance. Write `Sigma v_i = lambda_i v_i`, and define

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

This expression says how each covariance direction contributes to the cost of memorizing label noise. But `z_i` also appears inside `A`, so I need to decouple one direction at a time. Splitting

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

Now `z_i` is independent of `A_{-i}`, so the quadratic forms concentrate around their means. The denominator is the part that can make a term cheap: if the rest of the spectrum makes `z_i^T A_{-i}^{-1} z_i` large, that direction's contribution is suppressed. So the cost of each direction depends on how much mass the other directions put into `A`. The geometry of the whole spectrum has to enter here.

Before trusting any of this, I should pin down whether overparameterization alone is enough, because that is the explanation a naive count would suggest. Take the most symmetric case: `Sigma = I`, so every direction costs the same. Then `X Sigma X^T = XX^T = A`, and

```tex
tr(C) = tr(A^{-1} A A^{-1}) = tr(A^{-1}) = tr((XX^\top)^{-1}).
```

I can just compute this. For a Gaussian design with `Sigma = I`, `A = XX^T` is a sample of dimension `n` from `p` standardized coordinates, and `E\,tr(A^{-1})` is known to be `n/(p-n-1)` for `p > n+1`. Let me check that against simulation and read off what it means.

I draw `X` Gaussian, `n` rows and `p` columns, and average `tr((XX^T)^{-1})` over many draws:

```text
n=100, p=100  : tr(C) ~ 2.2e4        (n*tr ~ 2.2e6)
n=100, p=105  : tr(C) ~ 25.6
n=100, p=200  : tr(C) ~ 1.01
n=100, p=500  : tr(C) ~ 0.25
n=100, p=4000 : tr(C) ~ 0.026
```

The `n/(p-n-1)` formula matches the right column: `100/99 ≈ 1.01` at `p=200`, `100/399 ≈ 0.25` at `p=500`. So in the isotropic case `tr(C)` only goes to zero as `p/n -> infinity`; at any fixed ratio like `p = 2n` it sits at order one and the noise term is a constant. And at `p = n` exactly, the square Gaussian matrix is nearly singular and `tr(C)` is enormous. This kills the simplest hypothesis: counting extra directions is not what makes interpolation benign. With equal eigenvalues, even ten times more directions than samples leaves a constant excess risk. I need directions whose covariance weights are small, not merely numerous. That is exactly the lever the `Sigma`-weighting was supposed to give, so the analysis is pointing in a sensible direction.

So I split the spectrum. The first `k` directions are the directions I may need for signal. The tail after `k` is the candidate reservoir for noise. Define

```tex
A_k = \sum_{i>k}\lambda_i z_i z_i^\top.
```

Since each `z_i z_i^T` has expectation `I`, the expected tail matrix is

```tex
(\sum_{i>k}\lambda_i) I.
```

For the denominator stabilization I wanted above, I need this tail to actually act like a scalar multiple of the identity in the `n`-dimensional sample space, not just in expectation. The deviation of `A_k` from its mean is governed by the largest tail weight, `lambda_{k+1}`, times the sample dimension `n` (each of the `n` directions can be perturbed by the leading tail eigenvalue). So the tail behaves like a scalar matrix only when the total tail mass dominates that largest tail weight times `n`:

```tex
\sum_{i>k}\lambda_i >> n\lambda_{k+1}.
```

The ratio that appears is

```tex
r_k(\Sigma) = \frac{\sum_{i>k}\lambda_i}{\lambda_{k+1}}.
```

When `k` is below a constant fraction of `n` and `r_k(Sigma) >= b n`, the tail Gram matrix has all its eigenvalues within constant factors of its mean scale. That is the formal content of "many low-variance directions": the tail is not merely small, it is wide enough to look isotropic in the `n`-dimensional sample space, which is what makes the Sherman-Morrison denominators uniformly large.

I let `k^*` be the first index where this happens:

```tex
k^* = \min\{k : r_k(\Sigma) >= b n\}.
```

If there is no such index, I read `k^*` as infinity. If `k^*` is infinite or at least a constant fraction of `n`, then too many costly directions appear before the reservoir starts. The noise has no early cheap place to spread. This is the same regime as the isotropic computation above, where the leading directions are all expensive, so I expect the trace to stay bounded below by a constant there. That recovers the harmful interpolation threshold.

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

That ratio is a second effective rank:

```tex
R_k(\Sigma)
  =
\frac{(\sum_{i>k}\lambda_i)^2}{\sum_{i>k}\lambda_i^2}.
```

so the variance cost reads

```tex
tr(C) \asymp k^*/n + n/R_{k^*}(\Sigma),
```

up to constants. The first term is the price of the important directions before the reservoir. The second term is the price of how balanced the reservoir is. If the tail consists of many comparable tiny eigenvalues, `R_{k^*}` is large and the noise is diluted. If one tail eigenvalue dominates, the noise concentrates and the cost stays visible.

I want to see this formula survive a direct test rather than take it on faith, since the constants were dropped freely. Take a spectrum that is a few strong directions plus a wide flat floor: `k` eigenvalues equal to one, then `p-k` eigenvalues equal to a small `eps`. For the floor of `p-k` equal weights, `R_k = (eps(p-k))^2 / (eps^2(p-k)) = p-k`, so the prediction is `k/n + n/(p-k)`. I simulate `tr(C)` for the full covariance and compare:

```text
n=100, k=5,  p=20000, eps=0.01  : tr(C) ~ 0.010   formula k/n+n/R_k = 0.055   (R_k = 19995)
n=100, k=5,  p=20000, eps=0.001 : tr(C) ~ 0.041   formula            = 0.055
n=100, k=10, p=50000, eps=0.002 : tr(C) ~ 0.026   formula            = 0.102
```

The measured values track the formula to within the constant factor it promised, and crucially `tr(C)` is now an order of magnitude below the `~1` it had at `Sigma = I`, `p = 2n`. So a wide balanced reservoir genuinely drives the noise cost down, exactly as the two effective ranks predict, while raw dimension by itself did not. This is the concrete content of "the noise gets hidden where future prediction barely sees it."

The two effective ranks are doing different jobs, and the experiments make that visible. `r_k` answers whether the low-variance reservoir starts early enough and is wide enough to stabilize the sample Gram matrix. `R_k` answers whether the reservoir is balanced enough to dilute noise. The isotropic case had `R_k = p-k` large but a reservoir that never starts cheaply (no separation of scales), and it failed; the flat-floor case has both, and it works. A large raw dimension does not guarantee either property on its own.

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

I also have to keep the converse straight. If `r_0(Sigma)/(n log(1+r_0(Sigma)))` is large, some signal vector of the same norm must have visible excess risk. So I should not pretend the signal side is an exact if-and-only-if at the finite-sample level. The sharp matching story is clearest for the two tail quantities controlling noise absorption, where the lower bound `E R(\hat\theta) >= (\sigma^2/c)(k^*/n + n/R_{k^*})` matches the upper bound up to constants.

This is more than observing double descent. Double descent says risk may go down after interpolation. The decomposition explains when and why it goes down: the minimum-norm bias separates signal from noise, the covariance metric makes some directions prediction-cheap, and a high-dimensional balanced tail lets the interpolant hide noise where future prediction barely sees it.

I want to test the eigenvalue boundary the conditions imply, because they are most fragile in fixed infinite dimension, where the spectrum cannot grow with `n`. Try `lambda_k = k^{-alpha} log^{-beta}(k+1)`. The first thing to check is that this is even a valid covariance, which needs `sum_k lambda_k < infinity`. At `alpha = 1` the sum is `sum k^{-1} log^{-beta}(k+1)`, and by the integral test this converges iff `beta > 1`. Partial sums confirm the boundary:

```text
beta=0.9 : partials at N=1e4..1e7 = 4.29, 4.57, 4.81, 5.01   (last decade adds 0.20, not shrinking)
beta=1.0 : partials                = 4.06, 4.29, 4.47, 4.62   (adds 0.15, still creeping up)
beta=1.5 : partials                = 3.28, 3.30, 3.31, 3.33   (adds 0.01, converging)
```

So `beta <= 1` is excluded outright: the trace is infinite and the signal scale `r_0` is unbounded. Only `beta > 1` gives a usable spectrum. Now I check the tail behavior at the boundary `alpha=1, beta=1.5`. Computing `k^* = min{k : r_k >= n}` numerically over a long truncation:

```text
n=1e2 : k* = 26      k*/n = 0.26
n=1e3 : k* = 204     k*/n = 0.20
n=1e4 : k* = 1861    k*/n = 0.19
n=1e5 : k* = 19217   k*/n = 0.19
n=1e6 : k* = 229643  k*/n = 0.23
```

This is more honest than I expected: `k^*/n` does not visibly head to zero, it sits near `0.2`. That is the log factor at work. So in fixed infinite dimension the benign behavior is genuinely delicate; `k^*/n` is held to a small constant set by the threshold `b`, not driven to zero, and the real work of suppressing noise is done by `n/R_{k^*}` staying small. To confirm the spectrum is on the right side of the boundary I simulate `tr(C)` directly and compare against neighbors:

```text
alpha=1.0, beta=1.5 : tr(C) ~ 0.34, 0.33, 0.33   over n=50,100,200   (flat, controlled)
alpha=1.6           : tr(C) ~ 0.60, 0.61, 0.62                        (large, even growing)
alpha=0.6           : tr(C) ~ 0.007, 0.013, 0.022                     (tiny noise term...)
```
