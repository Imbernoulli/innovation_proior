I start with the fact that direct convolution is the wrong place to look for the general pattern. In the binomial case I can count coefficients and watch the middle of the expansion become smooth, but that computation is too special. If every summand has its own law, the sum is still a convolution, yet the density formula grows an extra integral with each new term and tells me less, not more. I need a representation in which independence simplifies the algebra instead of compounding it.

So I look for an object attached to a distribution that always exists and that turns addition of independent variables into multiplication. A bounded oscillatory exponential has exactly that feature. For a real variable `X`, the quantity `phi(t) = E exp(itX)` is defined for every real `t` with no integrability assumption, since `|exp(itX)| = 1`. And for independent `X, Y` the transform of the sum factors: `E exp(it(X+Y)) = E exp(itX) E exp(itY)`. That converts a convolution of many laws into a product of many numbers near `1`. I would much rather control a product of scalars than an iterated convolution.

The next question is what each factor looks like after normalization. Take the iid case first, centered with variance `1`, and form `S_n / sqrt(n)`. By the factorization its transform is `[phi(t/sqrt(n))]^n`. I expand `phi` near zero. Differentiating under the expectation, `phi(0)=1`, `phi'(0)=iE X=0` because the variable is centered, and `phi''(0)=-E X^2 = -1`. So `phi(t) = 1 - t^2/2 + o(t^2)`. Then `phi(t/sqrt(n)) = 1 - t^2/(2n) + o(1/n)`, and I want the limit of the `n`-th power.

Rather than assert what that power tends to, I take logarithms and actually compute. `log[phi(t/sqrt(n))]^n = n log(1 - t^2/(2n) + o(1/n))`. Using `log(1+u) = u + O(u^2)` with `u = -t^2/(2n)+o(1/n)`, the leading term is `n(-t^2/(2n)) = -t^2/2`, and the correction is `n O(1/n^2) = O(1/n) -> 0`. So the log tends to `-t^2/2` and the power tends to `exp(-t^2/2)`.

I want to see this happen on a real example, not just believe the asymptotics, so I pick `X` uniform on `{-1,+1}`, which has mean `0` and variance `1` and the clean transform `phi(t)=cos(t)`. I evaluate `[cos(t/sqrt(n))]^n` at `t=1.7`:

```
n=     1   -0.12884
n=     4    0.18973
n=    16    0.22521
n=    64    0.23317
n=   256    0.23510
n=  1024    0.23559
n=  8192    0.23573
exp(-t^2/2) = 0.23575
```

The values climb monotonically toward `exp(-(1.7)^2/2)=0.23575` and are already correct to four places by `n=8192`. So the quadratic-then-exponentiate mechanism is real arithmetic, not wishful thinking: the `-t^2/2` in the exponent is exactly the variance of `X` reappearing, and `Var(X)=1` is the only feature of the law that survived. Levy's continuity theorem turns the pointwise transform limit `exp(-t^2/2)`, which is continuous at `0`, back into convergence in distribution, and `exp(-t^2/2)` is the transform of `N(0,1)`. The iid case is done.

Now I want the general row of independent, non-identical summands `X_{n,1},...,X_{n,k_n}`, centered, with variances `sigma_{n,j}^2` summing to `1`. The transform of the sum is `prod_j E exp(itX_{n,j})`, and the obvious move is to compare each factor with the quadratic proxy `1 - t^2 sigma_{n,j}^2/2` and hope the product of proxies tends to `exp(-t^2/2)` the same way. Before trusting that, I should ask whether `sum_j sigma_{n,j}^2 -> 1` is by itself enough to force normality. If it is, there is no extra condition to find.

I test that hypothesis with a row designed to keep the variances summing to `1` while one term refuses to shrink. Let `k_n = n`. Put one "big" term `B_n = pm 1/sqrt(2)` (a sign-symmetric atom of variance `1/2`), and `n-1` "small" terms each `pm m` with `m^2 = 1/(2(n-1))`, so the small block also has total variance `1/2`. Total variance is exactly `1` for every `n`, so the normalization hypothesis holds perfectly. I simulate the sum at `n=4000` over `2*10^5` draws:

```
mean   -0.0002   var 0.9990
 -2.5 ##
 -2.0 ########
 -1.5 ####################
 -1.0 ##################################
 -0.6 #######################################
 -0.4 #######################################
 -0.1 ##########################################
  0.1 #######################################
  0.4 ########################################
  0.8 ##################################
  1.5 ####################
  2.0 ########
  2.5 ##
P(|S|<0.3) array = 0.2073  vs N(0,1) = 0.2358
```

The variance is `1` as required, but the shape is wrong. The `pm 1/sqrt(2)` atom holds mass away from the center, so the sum looks like a normal convolved with that two-point law, and the central probability `P(|S|<0.3)=0.207` falls clearly short of the normal value `0.236`. So total variance summing to `1` is necessary but not sufficient: a single summand that keeps a fixed fraction of the variance on a fixed-size scale survives the limit and spoils normality. The missing ingredient is exactly a condition that kills such a term.

That tells me where to put the extra hypothesis, and it tells me the form. In the failing array the offending mass is the part of each second moment that lives at absolute size bounded away from zero. So the condition I want is that for every fixed `epsilon > 0`, the total variance carried by the parts of the summands with `|X_{n,j}| > epsilon` tends to zero: `sum_j E[X_{n,j}^2 1{|X_{n,j}|>epsilon}] -> 0`. The big-term array violates this — its atom sits at `1/sqrt(2) > epsilon` for any `epsilon < 1/sqrt(2)`, contributing a fixed `1/2` to the sum no matter how large `n` is — which is consistent with its failure.

With that candidate condition I return to the factor-by-factor comparison and try to bound the accumulated error honestly. Everything rests on one inequality for the integrand: how far is `exp(ix)` from its two-term and three-term Taylor stubs. I need both stubs because the two regions of the split want different powers — the small region wants `x^2` so the cutoff helps, and the body wants `x^3/6` to beat the bounded variances. The clean statement is `|exp(ix) - (1 + ix - x^2/2)| <= min(|x|^3/6, x^2)`. I do not want to take this on faith, so I check it over `x in [-6,6]`:

```
max of  |e^{ix} - (1 + ix - x^2/2)| - min(|x|^3/6, x^2)  = 0.0
```
