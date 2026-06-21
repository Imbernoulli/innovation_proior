# Theorem

Let `x_1, ..., x_N in R^d` be linearly separable after absorbing labels into the examples, so there is `w_*` with `w_*^T x_n > 0` for all `n`. Let

```text
L(w) = sum_n ell(w^T x_n),
```

where `ell` is positive, differentiable, strictly decreasing to zero, beta-smooth, has `lim sup_{u -> -infinity} ell'(u) < 0`, and `-ell'(u)` has a tight exponential tail after absorbing its leading constants into the data scale and step size. Run fixed-step gradient descent

```text
w(t + 1) = w(t) - eta grad L(w(t)),
eta < 2 beta^{-1} sigma_max(X)^{-2}.
```

Then

```text
w(t) = what log t + rho(t),
```

where

```text
what = argmin_w ||w||^2
       subject to w^T x_n >= 1 for all n
```

is the hard-margin `L_2` SVM vector. For almost every dataset, `rho(t)` is bounded. For all separable datasets, the more complete expansion has only smaller iterated-log terms, so `rho(t) = O(log log t)` relative to the leading form.

Consequently,

```text
w(t) / ||w(t)|| -> what / ||what||.
```

# Rates

For almost every dataset,

```text
|| w(t)/||w(t)|| - what/||what|| || = O(1/log t).
```

For all datasets,

```text
|| w(t)/||w(t)|| - what/||what|| || = O(log log t / log t).
```

The normalized margin gap satisfies

```text
1/||what|| - min_n x_n^T w(t) / ||w(t)|| = O(1/log t),
```

while the training loss satisfies

```text
L(w(t)) = O(1/t).
```

Thus direction and margin converge logarithmically slowly even though the loss falls polynomially in time.

# Why The Bias Appears

For exponential loss,

```text
-grad L(w) = sum_n exp(-w^T x_n) x_n.
```

If `w = g(t) w_infty + rho(t)` and `g(t) -> infinity`, then points with the smallest value of `w_infty^T x_n` dominate the gradient; all larger-margin points are exponentially suppressed. The limiting direction, rescaled to unit minimal margin, must therefore be a nonnegative combination of the smallest-margin points:

```text
what = sum_n alpha_n x_n,
alpha_n >= 0 on support vectors,
alpha_n = 0 off support.
```

Together with `what^T x_n = 1` on support vectors and `what^T x_n > 1` off support, these are the KKT conditions for the hard-margin SVM. The exponential tail makes the support-vector set visible; full gradient descent supplies the Euclidean geometry, hence `L_2` max margin.

# Proof Spine

1. There is no finite critical point. For any finite `w`,

```text
w_*^T grad L(w) = sum_n ell'(w^T x_n) w_*^T x_n < 0.
```

Smooth GD with a small enough step gives `grad L(w(t)) -> 0`, so the norm and all margins diverge and the loss goes to zero at infinity.

2. The support-vector gradient suggests `what` and the growth law. On support vectors, `what^T x_n = 1`, so the leading gradient scale is `exp(-g(t))`. Matching this to the leading motion `g'(t)` gives `g(t) = log t`.

3. Define the residual

```text
r(t) = w(t) - what log t - wtilde,
eta exp(-x_n^T wtilde) = alpha_n,  n in S.
```

The offset makes

```text
eta sum_{n in S} exp(-x_n^T wtilde) x_n = what,
```

so the dangerous `what/t` term cancels.

4. The support-vector part is nonpositive because `z(exp(-z)-1) <= 0`; non-support vectors decay like `t^{-theta}` for `theta = min_{n not in S} what^T x_n > 1`; logistic-tail and discrete-step errors are summable. Hence the residual is bounded generically. Degenerate zero-dual support vectors are handled by recursively adding smaller `log log t` terms.

# Practical Consequence

Training loss can be nearly zero long before the classifier direction is close to its limiting large-margin separator. Continuing to optimize can still improve margin and classification error. Validation loss can even rise like `Omega(log t)` if the final direction misclassifies a validation point, because the growing norm amplifies that point's negative margin.
