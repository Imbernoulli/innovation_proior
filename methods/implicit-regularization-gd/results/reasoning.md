I start with the obstruction, because it tells me what kind of answer is even possible. In least squares I can talk about a finite interpolating solution, and gradient descent has a clean row-space story that ends at the minimum Euclidean norm interpolant. Here that story breaks immediately. On separable classification data, logistic loss and exponential loss do not have a finite minimizer. If a vector separates the data, scaling it up keeps decreasing the loss. So I cannot ask which finite minimizer gradient descent selects. The norm is going to infinity.

What remains is direction. The prediction only depends on the sign of `w^T x`, so if anything converges it has to be `w(t)/||w(t)||`. I need to understand a trajectory that runs away to infinity and still settles in direction.

First I check that the trajectory really has to run away. Let the labels be absorbed into the examples, so separability gives me some `w_*` with `w_*^T x_n > 0` for all `n`. For any finite `w`,

```text
w_*^T grad L(w) = sum_n ell'(w^T x_n) w_*^T x_n.
```

Every term is negative because `ell' < 0` and `w_*^T x_n > 0`. Therefore no finite critical point exists. But the smooth-descent inequality for a small enough fixed step size gives `sum_t ||grad L(w(t))||^2 < infinity`, hence `grad L(w(t)) -> 0`. With no finite critical point, the only way this can happen is that the margins go to infinity and the loss goes to zero at infinity. So the direction question is not cosmetic; it is the only possible question.

Now I look at the gradient in the simplest case, the exponential loss. It is

```text
-grad L(w) = sum_n exp(-w^T x_n) x_n.
```

Suppose the direction is beginning to look like some vector `w_infty`, and write `w = g(t) w_infty + rho(t)` with `g(t)` large. Then each example is weighted by

```text
exp(-g(t) w_infty^T x_n) exp(-rho(t)^T x_n).
```

The smallest value of `w_infty^T x_n` gives the least negative exponent. As `g(t)` grows, every point with larger margin is exponentially suppressed relative to these smallest-margin points. The gradient stops seeing the whole dataset equally. It concentrates on the points closest to the separating hyperplane.

That concentration is the missing mechanism. If the late gradient is made from the smallest-margin points, then the limiting direction itself must be a nonnegative combination of those points, because the finite initialization is negligible compared with the diverging norm. I rescale the candidate limit so its smallest margin is one and call it `what`. Then I get

```text
what = sum_n alpha_n x_n,
```

with `alpha_n >= 0` only on points satisfying `what^T x_n = 1`, and `alpha_n = 0` on points satisfying `what^T x_n > 1`. These are not just suggestive conditions. They are the KKT conditions for

```text
min_w ||w||^2
subject to w^T x_n >= 1 for all n.
```

So the asymptotic gradient points me at the hard-margin `L_2` SVM direction. The loss tail selects the support vectors; the Euclidean update geometry of full gradient descent selects the `L_2` max-margin separator.

This is already more specific than saying optimization affects generalization. The older regularization-path result says that explicit `L_p` penalties, sent to zero, approach `L_p` max-margin directions. Boosting results say coordinate descent has `L_1` margin behavior. Those are important clues, but they are not this trajectory. I am not adding a vanishing penalty, and I am not taking coordinate steps. I am asking what unregularized full-gradient descent does while its norm diverges.

I still have to make the argument self-consistent. If the leading direction is `what`, how fast does its coefficient grow? On a support vector, `what^T x_n = 1`. If `w(t) = what g(t) + rho(t)`, then the support-vector gradient is on the order of `exp(-g(t))`. In continuous time the leading motion is `what g'(t)`. Matching the two asks for

```text
g'(t) ~= exp(-g(t)).
```

This integrates to `g(t) ~= log t`. That is why the leading term is not linear in `t` or a power of `t`; the gradient shrinks exponentially in the margin, and the fixed point of that feedback is logarithmic growth.

Now I need a residual proof. I define

```text
r(t) = w(t) - what log t - wtilde.
```

The offset `wtilde` is not decoration. It is chosen so that the leading support-vector gradient exactly reconstructs the SVM vector:

```text
eta exp(-x_n^T wtilde) = alpha_n,  n in S.
```

Then

```text
eta sum_{n in S} exp(-x_n^T wtilde) x_n = what.
```

This means the `what/t` term coming from the derivative of `what log t` cancels the `1/t` support-vector part of the gradient.

In continuous time with exact exponential loss, the residual calculation is almost bare. The derivative of `||r||^2/2` splits into support-vector terms and non-support-vector terms. The support-vector contribution becomes

```text
(1/t) sum_{n in S} exp(-x_n^T wtilde)
       (exp(-x_n^T r) - 1)(x_n^T r).
```

For every real `z`, `z(exp(-z)-1) <= 0`. So the support vectors do not push the residual outward; after the cancellation, they damp it. The non-support vectors have `what^T x_n > 1`. If

```text
theta = min_{n not in S} what^T x_n,
```

then `theta > 1`, and their terms are bounded by a multiple of `t^{-theta}`. That is summable. A nonpositive support part plus a summable interior-point part means `||r(t)||` stays bounded.

The discrete logistic proof has to pay for the facts that `log(t+1)-log t` is only approximately `1/t` and that the logistic derivative is only asymptotically exponential. But the shape is the same. The squared residual step expands as

```text
||r(t+1)||^2
= ||r(t+1)-r(t)||^2
  + 2(r(t+1)-r(t))^T r(t)
  + ||r(t)||^2.
```

The first new term is summable because the smooth-descent lemma gives `sum_t ||grad L(w(t))||^2 < infinity` and because `log^2(1+1/t)` is bounded by `t^{-2}`. The second term is the real work. I use the tight exponential-tail bounds on `-ell'(u)` and split support-vector cases by the sign and size of `x_n^T r(t)`. The leading exponential part again gives the favorable `z(exp(-z)-1)` sign. The tail errors are smaller powers such as `t^{-1-1.5 mu_+}` and `t^{-1-0.5 mu_-}`, while non-support points still give `t^{-theta}`. The combined bound is

```text
(r(t+1)-r(t))^T r(t)
<= C t^{-min(theta, 1+1.5 mu_+, 1+0.5 mu_-)}.
```

All exponents are larger than one, so the accumulated error is finite and the residual is bounded.

There is also a sharper fact hiding inside the same proof. If the support-vector component of the residual is not small, at least one support vector has `|x_n^T r(t)|` bounded away from zero. Then the support-vector term is not merely nonpositive; it is at most `-C/t`. If the support vectors span the data, the off-span residual is frozen by the dynamics and can be absorbed into `wtilde`. The recurring `-C/t` drift then forces the residual to converge to zero; otherwise summing `-C/t` would eventually make `||r||^2` negative, impossible.

I have to account for degeneracy. The clean offset equation needs positive dual coefficients on the support vectors. Generically this is true. For a fixed support set, the KKT equations give

```text
alpha_S = (X_S^T X_S)^{-1} 1.
```

Each coefficient is a rational function of the data. A coefficient vanishes only on a polynomial zero set, except in the impossible case where that polynomial is identically zero. So zero support-vector coefficients occur only on measure-zero datasets.

When such a zero coefficient does occur, the finite offset cannot absorb everything. The proof then recurses: take the zero-coefficient support vectors, project away the span already explained by positive-coefficient support vectors, and solve another max-margin problem in the remaining subspace. This adds smaller iterated-log terms:

```text
w(t) = sum_{m=1}^M what_m log^{circ m}(t) + rho(t).
```

The first term is still `what_1 log t`, where `what_1` is the original hard-margin direction, so the normalized direction still converges to the SVM direction. The price is a possible `log log t` factor in the direction rate.

Now the rates fall out of normalization. In the generic case,

```text
w(t) = what log t + rho(t)
```

with bounded `rho(t)`. Dividing by the norm and expanding `1/sqrt(1+x)` gives

```text
w(t)/||w(t)||
= what/||what||
  + (I - what what^T / ||what||^2) rho(t) / (||what|| log t)
  + O(1/log^2 t).
```

Only the residual component perpendicular to `what` matters to first order. Therefore the direction converges as `O(1/log t)` for almost every dataset and as `O(log log t / log t)` in the degenerate all-dataset statement. The margin gap is `O(1/log t)`. The loss, however, is dominated by support vectors:

```text
L(w(t)) = (1/t) sum_{n in S} exp(-rho(t)^T x_n) + smaller terms,
```

so it is `O(1/t)`.

This separation of rates explains why the theorem matters. The loss can look essentially finished while the classifier direction is still moving slowly toward the large-margin separator. To get an `epsilon` direction error, I need roughly `log t` on the order of `1/epsilon`, so the loss has already become exponentially small in `1/epsilon`. Continuing optimization past zero training error is not just polishing a numerical objective; it keeps changing the normalized predictor.

It also explains the validation-loss paradox. If the limiting training separator misclassifies a validation point, then the norm growth makes that point's logistic loss grow like `log t`, even while margins and classification error can improve. So validation loss can rise for a reason that is not ordinary overfitting. The correct observable for this asymptotic story is classification error or margin behavior, not loss alone.

The final result I arrive at is this: for separable homogeneous linear classification with a smooth monotone loss whose negative derivative has a tight exponential tail, fixed-step gradient descent with a small enough step size satisfies `w(t) = what log t + rho(t)`, where `what` is the hard-margin `L_2` SVM solution; the residual is bounded generically and grows at most like `log log t` in degenerate cases; therefore the normalized iterate converges to the `L_2` max-margin direction. The reason is the exponential tail's support-vector concentration plus the `log t` cancellation that keeps the residual bounded. The insight is not merely that optimization matters; it is that the asymptotic gradient has an invariant support-vector geometry that exposes the optimizer's hidden regularizer.
