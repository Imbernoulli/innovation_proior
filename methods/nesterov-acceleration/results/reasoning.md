I start with the one fact the oracle gives me reliably. At a point `y`, smoothness says the function is below a quadratic model, so the step `y-(1/L)grad f(y)` buys at least `(1/(2L))||grad f(y)||^2` of function decrease. If I keep taking that step at the current iterate, I get the familiar proof: the objective gap decreases by a squared-gradient term, convexity turns the gap into a gradient-distance product, and the distance to `x*` does not increase. The result is `f(x_t)-f* <= 2L||x_1-x*||^2/(t-1)`. This proof is clean, but it is also revealingly small. It spends one local upper model, gets one local decrease, and forgets almost everything except the new point. If the lower bound for this oracle class were also `1/t`, that would be the end of the story.

But the hard examples do not say that. A tridiagonal quadratic can be arranged so the first gradient exposes only the first coordinate, the next gradient can expose only the next coordinate, and after `t` oracle calls every black-box span method is still blind to a long tail of the minimizer. Computing the residual tail gives a floor of order `L R^2/t^2` for smooth convex functions. In the strongly convex version, the hidden tail is geometric with ratio `(sqrt(kappa)-1)/(sqrt(kappa)+1)`, so the floor is on the `sqrt(kappa)` scale. That changes the question. Gradient descent is not merely missing a constant; its proof is too weak by a whole factor of `t`, and in the strongly convex case by a square root of the condition number.

The quadratic methods tempt me in the right direction. Momentum can carry an iterate through a narrow valley, and conjugate gradients exploit polynomial filtering of the Hessian. On a quadratic, the recurrence decomposes by eigenvalue and the `sqrt(kappa)` scale appears naturally. But that explanation is spectral. Once the function is no longer quadratic, the gradient at the momentum-shifted point is no longer governed by a fixed matrix, and an inertial step taken at the current point gives me no global quantity that must improve. I need the square-root geometry without a quadratic-only proof.

So I stop trying to make every step a descent step. The primary source explicitly allows the sequence not to be relaxational, and this is the important permission. If the function value can rise on some iterations, the certificate cannot be "the last step decreased enough." It has to be a global certificate that tightens across all oracle calls.

Every gradient call gives me a global lower model. For convex `f`, the line `f(y)+<grad f(y),x-y>` sits below `f(x)` for every `x`; with strong convexity I can add `(mu/2)||x-y||^2`. Instead of throwing these lower models away, I try to average them into a simple model I can track. I want functions `phi_k` and weights `lambda_k -> 0` such that

`phi_k(x) <= (1-lambda_k)f(x) + lambda_k phi_0(x)` for all `x`.

If I can also keep `f(x_k) <= phi_k^*`, where `phi_k^* = min_x phi_k(x)`, then at the optimum

`f(x_k)-f* <= lambda_k(phi_0(x*)-f*)`.

Now the rate is not whatever a local descent proof can squeeze from one step. The rate is exactly how fast I can make `lambda_k` shrink while maintaining the inequality `f(x_k) <= phi_k^*`.

I choose `phi_0` as a quadratic, `phi_0(x)=phi_0^*+(gamma_0/2)||x-v_0||^2`, and update by mixing in the lower model at a point `y_k`:

`phi_{k+1}(x)=(1-alpha_k)phi_k(x)+alpha_k[f(y_k)+<grad f(y_k),x-y_k>+(mu/2)||x-y_k||^2]`,

with `lambda_{k+1}=(1-alpha_k)lambda_k`. This preserves the estimating-sequence inequality because the bracket is always a lower bound on `f`. It also preserves a simple quadratic form:

`phi_k(x)=phi_k^*+(gamma_k/2)||x-v_k||^2`,

where `gamma_{k+1}=(1-alpha_k)gamma_k+alpha_k mu` and

`v_{k+1}=[(1-alpha_k)gamma_k v_k+alpha_k mu y_k-alpha_k grad f(y_k)]/gamma_{k+1}`.

Now the method has to be hidden in the maintenance condition. Suppose I already have `f(x_k) <= phi_k^*`. When I expand the recurrence for `phi_{k+1}^*`, use convexity to replace `phi_k^*` by a lower bound at `y_k`, and drop the nonnegative strong-convexity piece, I get a lower bound with two dangerous terms:

`phi_{k+1}^* >= f(y_k) - (alpha_k^2/(2 gamma_{k+1}))||grad f(y_k)||^2 + (1-alpha_k)<grad f(y_k), (alpha_k gamma_k/gamma_{k+1})(v_k-y_k)+x_k-y_k>`.

The first dangerous term has the shape of a gradient-step decrease. I can make it exactly match the smoothness inequality if I set

`L alpha_k^2 = gamma_{k+1}`.

Then the point

`x_{k+1}=y_k-(1/L)grad f(y_k)`

satisfies `f(x_{k+1}) <= f(y_k) - (1/(2L))||grad f(y_k)||^2`, so the gradient-norm part is paid for.

The second dangerous term is worse because it has no sign. I cannot bound it. I have to remove it. That forces

`(alpha_k gamma_k/gamma_{k+1})(v_k-y_k)+x_k-y_k=0`,

so

`y_k=(alpha_k gamma_k v_k+gamma_{k+1}x_k)/(gamma_k+alpha_k mu)`.

This is the decisive point. The gradient is not read at the current iterate and then decorated with momentum. The proof forces me to form a look-ahead point, a carefully chosen blend of the current output point and the minimizer of the accumulated lower model, and to read the gradient there. Momentum appears only after I eliminate the hidden model variables. Its coefficient is not an inertial guess; it is the value that cancels the cross term in the global certificate.

With that choice, the induction closes: `f(x_{k+1}) <= phi_{k+1}^*`. The convergence rate is now controlled by `lambda_k=prod_i(1-alpha_i)`. In the merely convex case `mu=0`, the relation `L alpha_k^2=gamma_{k+1}` and the quadratic recurrence make the reciprocal coefficient grow linearly. In the usual two-sequence notation this is

`a_0=1`, `a_{k+1}=(1+sqrt(1+4a_k^2))/2`, and the momentum coefficient is `(a_k-1)/a_{k+1}`.

Since `a_k` grows like `k/2`, the weight `lambda_k` falls like `1/k^2`. That gives the accelerated smooth-convex rate and matches the tridiagonal lower-bound order.

In the strongly convex case there is a fixed point. If I start with `gamma_0=mu`, then `gamma_k=mu` and the equation `L alpha^2=mu` gives `alpha=sqrt(mu/L)=1/sqrt(kappa)`. The two-sequence coefficient becomes

`(1-alpha)/(1+alpha) = (sqrt(kappa)-1)/(sqrt(kappa)+1)`,

and `lambda_k=(1-alpha)^k`, so the rate has exponent `1/sqrt(kappa)`. This is the same square-root scale that appeared in the quadratic methods and in the hard lower bound, but now it is certified by accumulated lower models rather than by diagonalizing a Hessian.

The 1983 potential proof is the same accounting in a more compressed form. It defines `p_k=(a_k-1)(x_{k-1}-x_k)` and constructs a quantity involving `a_k^2(f(x_k)-f*)+||p_k-x_k+x*||^2`. The line-search inequality, convexity, and the identity `a_{k+1}^2-a_{k+1}=a_k^2` make the potential telescope. That proof lands directly on `C/(k+2)^2`; the later estimate-sequence view explains why the odd-looking extrapolation is the only choice that keeps the global certificate intact.

So the method is not "take gradient descent and add speed." It is a coupling between a point sequence and a hidden lower-bound sequence. The extrapolated point is where the next lower model must be sampled so that the model minimum stays above the next function value. The momentum coefficient is the visible trace of that coupling. This is why the sequence can be nonmonotone and still optimal: descent of the function value is replaced by descent of a certificate whose weight shrinks at the lower-bound rate.
