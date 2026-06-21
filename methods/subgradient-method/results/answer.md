# The Subgradient Method

For convex `f : R^n -> R`, choose any `g_k in partial f(x_k)` and update

```text
x_{k+1} = x_k - alpha_k g_k.
```

Because `-g_k` need not be a descent direction, track and return the best value seen:

```text
f_best^(k) = min_{1 <= i <= k} f(x_i).
```

## Guarantee

Assume `x*` minimizes `f`, `f*=f(x*)`, `||x_1-x*|| <= R`, and `||g_k|| <= G`. The subgradient inequality at `x*` gives

```text
g_k^T (x_k-x*) >= f(x_k)-f*.
```

Therefore

```text
||x_{k+1}-x*||^2
 <= ||x_k-x*||^2 - 2 alpha_k(f(x_k)-f*) + alpha_k^2 ||g_k||^2.
```

Telescoping yields

```text
f_best^(k)-f*
 <= (R^2 + G^2 sum_{i=1}^k alpha_i^2) / (2 sum_{i=1}^k alpha_i).
```

Step-size consequences:

- `sum alpha_k = infinity` and `sum alpha_k^2 < infinity` gives `f_best^(k) -> f*`.
- Constant `alpha` gives an asymptotic error floor `G^2 alpha/2`.
- Fixed horizon `K`: `alpha_i=(R/G)/sqrt(K)` gives `f_best^(K)-f* <= RG/sqrt(K)`.
- Polyak step, when `f*` is known: `alpha_k=(f(x_k)-f*)/||g_k||^2`.

Nesterov's resisting-oracle lower bound for nonsmooth Lipschitz convex minimization is `Omega(MR/sqrt(k))`, so the fixed-horizon `O(RG/sqrt(k))` guarantee is worst-case optimal up to constants.

## Code

See `code/subgradient_method.py` for an executable implementation with square-summable, fixed-horizon, Polyak, and estimated-Polyak step rules, plus a piecewise-linear `max_i(a_i^T x+b_i)` oracle example.
