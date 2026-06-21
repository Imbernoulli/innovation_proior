# Nesterov's Accelerated Gradient Method

For a convex differentiable objective with `L`-Lipschitz gradient, keep two sequences: `y_k`, where the gradient is evaluated, and `x_k`, the gradient-stepped point.

## Smooth Convex Case

Initialize `x_0=y_0` and `a_0=1`. Iterate

```
x_{k+1} = y_k - (1/L) grad f(y_k)
a_{k+1} = (1 + sqrt(1 + 4 a_k^2))/2
y_{k+1} = x_{k+1} + ((a_k - 1)/a_{k+1})(x_{k+1} - x_k)
```

The original line-search version replaces `1/L` by a halved step satisfying

`f(y_k)-f(y_k-alpha_k grad f(y_k)) >= (alpha_k/2)||grad f(y_k)||^2`.

Rates:

`f(x_k)-f* <= C/(k+2)^2`,

with `C=4L||y_0-x*||^2` under the 1983 line search and `C=2L||y_0-x*||^2` when `L` is known. The estimate-sequence form gives the comparable bound

`f(x_k)-f* <= 8L||x_0-x*||^2/(3(k+1)^2)`.

## Strongly Convex Case

If `f` is `mu`-strongly convex and `kappa=L/mu`, use the fixed coefficient

`theta = (sqrt(kappa)-1)/(sqrt(kappa)+1)`.

Then

```
x_{k+1} = y_k - (1/L) grad f(y_k)
y_{k+1} = x_{k+1} + theta (x_{k+1} - x_k)
```

and

`f(x_k)-f* <= ((L+mu)/2)||x_0-x*||^2 exp(-k sqrt(mu/L))`.

The 1983 strongly convex option restarts the smooth-convex scheme every `O(sqrt(L/mu))` iterations, halving the residual each block.

## Why This Is The Right Coupling

The proof builds estimating sequences

`phi_k(x) <= (1-lambda_k)f(x) + lambda_k phi_0(x)`.

If `f(x_k) <= min_x phi_k(x)`, then `f(x_k)-f* <= lambda_k(phi_0(x*)-f*)`. The update averages in the global lower model at `y_k`. Maintaining the inequality forces two choices:

`L alpha_k^2 = gamma_{k+1}`,

so the estimate-sequence gain matches the smoothness descent gain, and

`y_k=(alpha_k gamma_k v_k + gamma_{k+1}x_k)/(gamma_k+alpha_k mu)`,

so the unsigned cross term vanishes. Eliminating the hidden variables `v_k,gamma_k` gives the two-sequence momentum form above. The acceleration comes from this global certificate, not from monotone descent.

## Code

See `code/accelerated_gradient.py` for a minimal NumPy implementation of the convex and strongly convex schemes.
