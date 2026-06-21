# The Bayesian Occam Factor

For a model `H` with parameters `w`, compare models by the probability the whole model assigns to the observed data:

```text
P(D|H) = int P(D|w,H) P(w|H) dw
P(H|D) proportional to P(D|H) P(H)
```

With comparable model priors, rank by `P(D|H)`. This quantity is the normalizing constant of the parameter posterior, but it is the data-dependent term at the model-comparison level.

## Laplace Decomposition

If the posterior is peaked at `w_MP` and

```text
A = - nabla nabla log P(w|D,H) at w_MP,
```

then Laplace's method gives

```text
P(D|H) ~= P(D|w_MP,H) P(w_MP|H) (2*pi)^(k/2) |A|^(-1/2).
```

So

```text
evidence ~= best-fit likelihood * Occam factor
Occam factor = P(w_MP|H) (2*pi)^(k/2) |A|^(-1/2).
```

For one parameter with an approximately uniform prior width `sigma_w`,

```text
Occam factor = sigma_w|D / sigma_w.
```

It is the posterior accessible parameter volume divided by the prior accessible volume: the fraction of the model's parameter space that survives the data.

## Quadratic Interpolation Case

For Gaussian noise and a quadratic regularizer,

```text
P(D|w,beta,H) = exp(-beta E_n) / Z_n(beta)
P(w|alpha,H) = exp(-alpha E_w) / Z_w(alpha)
M(w) = alpha E_w + beta E_n
Z_M(alpha,beta) = int exp(-M(w)) dw
```

The evidence for the regularization and noise scales is

```text
P(D|alpha,beta,H) = Z_M(alpha,beta) / (Z_w(alpha) Z_n(beta)).
```

For quadratic `E_w,E_n`, with `A=alpha C + beta B`,

```text
log P(D|alpha,beta,H)
  = -alpha E_w^MP - beta E_n^MP
    - 0.5 log det A - log Z_w(alpha) - log Z_n(beta)
    + (k/2) log(2*pi).
```

The volume ratio term is the regularization-level Occam penalty.

## Effective Parameters

In the whitened prior basis where `C=I` and `A=alpha I + beta B`, let `lambda_a` be the eigenvalues of `beta B`. Maximizing the evidence gives

```text
gamma = k - alpha Tr(A^-1)
      = sum_a lambda_a / (lambda_a + alpha)
```

and

```text
2 alpha E_w^MP = gamma
2 beta E_n^MP = N - gamma.
```

`gamma` is the effective number of parameters measured by the data. It replaces both raw parameter count `k` and the discrepancy target `N`: the fitted data misfit satisfies `chi_D^2 = N - gamma`.

## Model Ranking

For a whole model family,

```text
P(D|H) = int P(D|alpha,beta,H) P(alpha,beta|H) d alpha d beta.
```

When this surface has a sharp maximum at `alpha_hat,beta_hat`,

```text
P(D|H) ~= P(D|alpha_hat,beta_hat,H)
         P(alpha_hat,beta_hat|H)
         2*pi * Delta log alpha * Delta log beta.
```

Ranking by this integrated evidence produces the characteristic trade-off: too-simple models lose by data misfit; over-flexible models lose by unused parameter volume.
