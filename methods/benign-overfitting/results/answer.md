# Benign Overfitting in Linear Regression

## Setup

In the well-specified linear model, let

```tex
y = x^\top\theta^* + \epsilon,
\qquad
E[\epsilon \mid x] = 0,
\qquad
\Sigma = E[xx^\top].
```

Assume `x = V Lambda^{1/2} z`, where the coordinates of `z` are independent unit-variance subgaussian variables, and assume the design has full row rank so interpolation is possible. Among all interpolating least-squares solutions, choose the minimum-norm one:

```tex
\hat\theta = X^\top(XX^\top)^{-1}y.
```

The excess risk is

```tex
R(\hat\theta)
  = E_{x,y}[(y-x^\top\hat\theta)^2 - (y-x^\top\theta^*)^2]
  = (\theta^*-\hat\theta)^\top\Sigma(\theta^*-\hat\theta).
```

## Core Decomposition

Let

```tex
P = X^\top(XX^\top)^{-1}X,
B = (I-P)\Sigma(I-P),
C = (XX^\top)^{-1}X\Sigma X^\top(XX^\top)^{-1}.
```

Then

```tex
R(\hat\theta) \le 2{\theta^*}^\top B\theta^* + 2\epsilon^\top C\epsilon,
```

and

```tex
E_\epsilon R(\hat\theta)
  \ge {\theta^*}^\top B\theta^* + \sigma^2 tr(C).
```

`B` is the cost of signal directions missed by the sample. `tr(C)` is the cost of exactly fitting the training noise.

## Spectral Quantities

For covariance eigenvalues `lambda_1 >= lambda_2 >= ...`, define

```tex
r_k(\Sigma) =
\frac{\sum_{i>k}\lambda_i}{\lambda_{k+1}},
\qquad
R_k(\Sigma) =
\frac{(\sum_{i>k}\lambda_i)^2}{\sum_{i>k}\lambda_i^2}.
```

Let

```tex
k^* = \min\{k >= 0 : r_k(\Sigma) >= b n\}.
```

The minimum is `infinity` if no such `k` exists. `r_k` locates a large enough low-variance tail for the tail Gram matrix to behave nearly isotropically. `R_k` measures how balanced that tail is, and therefore how evenly the memorized noise is diluted.

## Main Characterization

For constants depending only on the subgaussian norm, and for `log(1/delta) < n/c`, if `k^* >= n/c_1`, including `k^* = infinity`, then interpolation is harmful:

```tex
E R(\hat\theta) >= \sigma^2/c.
```

If `k^* < n/c_1`, then with high probability,

```tex
R(\hat\theta)
  <= c ||\theta^*||^2 ||\Sigma||
     max{ sqrt(r_0(\Sigma)/n), r_0(\Sigma)/n, sqrt(log(1/\delta)/n) }
   + c log(1/\delta)\sigma_y^2
     ( k^*/n + n/R_{k^*}(\Sigma) ),
```

and the noise term is necessary up to constants:

```tex
E R(\hat\theta)
  >= (\sigma^2/c)( k^*/n + n/R_{k^*}(\Sigma) ).
```

With `||Sigma_n||` normalized to one, a covariance sequence is benign under the sufficient conditions

```tex
r_0(\Sigma_n)/n -> 0,
\qquad
k_n^*/n -> 0,
\qquad
n/R_{k_n^*}(\Sigma_n) -> 0.
```

A separate lower bound for the signal term says that, for Gaussian covariates, some `theta^*` with a prescribed norm has constant excess risk when `r_0(Sigma)/(n log(1+r_0(Sigma)))` is large. Thus the signal condition is necessary up to that logarithmic gap; the two tail conditions above are the matching spectral conditions for the noise term.

## What The Result Explains

Exact interpolation can be low-risk because the covariance metric separates prediction-important directions from prediction-cheap directions. The minimum-norm interpolant learns the signal in the important directions while spreading the label noise through many low-variance directions.

This goes beyond empirical double descent. Double descent describes a risk curve after capacity crosses the interpolation threshold. The theorem gives a finite-sample, two-sided spectral reason for when the second descent is possible: the signal scale must be controlled, and the low-variance covariance tail must start early and be balanced enough to absorb noise.

The eigenvalue examples show the boundary. In fixed infinite dimension, benign behavior is narrow: for `lambda_k = k^{-alpha} log^{-beta}(k+1)`, it occurs exactly at `alpha = 1`, `beta > 1`. In growing finite dimension with a small isotropic floor, it is much broader: a large flat reservoir of weak directions can make overparameterized exact fitting harmless even when the main signal eigenvalues decay rapidly.
