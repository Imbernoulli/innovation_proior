I start with the one fact the oracle gives me reliably. At a point `y`, smoothness says the function is below a quadratic model, so the step `y-(1/L)grad f(y)` buys at least `(1/(2L))||grad f(y)||^2` of function decrease. If I keep taking that step at the current iterate, I get the familiar proof: the objective gap decreases by a squared-gradient term, convexity turns the gap into a gradient-distance product, and the distance to `x*` does not increase. The result is `f(x_t)-f* <= 2L||x_1-x*||^2/(t-1)`. This proof is clean, but it is also revealingly small. It spends one local upper model, gets one local decrease, and forgets almost everything except the new point. If the lower bound for this oracle class were also `1/t`, that would be the end of the story.

But the hard examples do not say that. A tridiagonal quadratic can be arranged so the first gradient exposes only the first coordinate, the next gradient can expose only the next coordinate, and after `t` oracle calls every black-box span method is still blind to a long tail of the minimizer. Computing the residual tail gives a floor of order `L R^2/t^2` for smooth convex functions. In the strongly convex version, the hidden tail is geometric with ratio `(sqrt(kappa)-1)/(sqrt(kappa)+1)`, so the floor is on the `sqrt(kappa)` scale. That changes the question. Gradient descent is not merely missing a constant; its proof is too weak by a whole factor of `t`, and in the strongly convex case by a square root of the condition number.

The quadratic methods tempt me in the right direction. Momentum can carry an iterate through a narrow valley, and conjugate gradients exploit polynomial filtering of the Hessian. On a quadratic, the recurrence decomposes by eigenvalue and the `sqrt(kappa)` scale appears naturally. But that explanation is spectral. Once the function is no longer quadratic, the gradient at the momentum-shifted point is no longer governed by a fixed matrix, and an inertial step taken at the current point gives me no global quantity that must improve. I need the square-root geometry without a quadratic-only proof.

So I stop trying to make every step a descent step. The standard convergence story insists the function value drop every iteration, but nothing in the oracle-complexity question requires that. If I am willing to let the value rise on some iterations, the certificate can no longer be "the last step decreased enough." It has to be a global certificate that tightens across all oracle calls. That is a real change of bookkeeping, so I want to keep in mind that whatever I build must still be checkable on an actual run where the value is non-monotone.

Every gradient call gives me a global lower model. For convex `f`, the line `f(y)+<grad f(y),x-y>` sits below `f(x)` for every `x`; with strong convexity I can add `(mu/2)||x-y||^2`. Instead of throwing these lower models away, I try to average them into a simple model I can track. I want functions `phi_k` and weights `lambda_k -> 0` such that

`phi_k(x) <= (1-lambda_k)f(x) + lambda_k phi_0(x)` for all `x`.

If I can also keep `f(x_k) <= phi_k^*`, where `phi_k^* = min_x phi_k(x)`, then evaluating the first inequality at `x*` gives `phi_k^* <= phi_k(x*) <= (1-lambda_k)f* + lambda_k phi_0(x*)`, so

`f(x_k)-f* <= lambda_k(phi_0(x*)-f*)`.

Now the rate is not whatever a local descent proof can squeeze from one step. The rate is exactly how fast I can make `lambda_k` shrink while maintaining the inequality `f(x_k) <= phi_k^*`.

I choose `phi_0` as a quadratic, `phi_0(x)=phi_0^*+(gamma_0/2)||x-v_0||^2`, and update by mixing in the lower model at a point `y_k`:

`phi_{k+1}(x)=(1-alpha_k)phi_k(x)+alpha_k[f(y_k)+<grad f(y_k),x-y_k>+(mu/2)||x-y_k||^2]`,

with `lambda_{k+1}=(1-alpha_k)lambda_k`. This preserves the estimating-sequence inequality because the bracket is always a lower bound on `f`. It also preserves a simple quadratic form: a convex combination of a quadratic and a quadratic is again a quadratic, so

`phi_k(x)=phi_k^*+(gamma_k/2)||x-v_k||^2`,

where matching the second-order coefficient gives `gamma_{k+1}=(1-alpha_k)gamma_k+alpha_k mu` and matching the first-order coefficient gives

`v_{k+1}=[(1-alpha_k)gamma_k v_k+alpha_k mu y_k-alpha_k grad f(y_k)]/gamma_{k+1}`.

The remaining freedom is in `y_k` and in the relation between `alpha_k` and `L`. They are pinned by the maintenance condition. Suppose I already have `f(x_k) <= phi_k^*`. When I expand the recurrence for `phi_{k+1}^*`, use convexity to replace `phi_k^*` by a lower bound at `y_k`, and drop the nonnegative strong-convexity piece, I get a lower bound with two dangerous terms:

`phi_{k+1}^* >= f(y_k) - (alpha_k^2/(2 gamma_{k+1}))||grad f(y_k)||^2 + (1-alpha_k)<grad f(y_k), (alpha_k gamma_k/gamma_{k+1})(v_k-y_k)+x_k-y_k>`.

The first dangerous term has the shape of a gradient-step decrease. I can make it exactly match the smoothness inequality if I set

`L alpha_k^2 = gamma_{k+1}`,

because then `alpha_k^2/(2 gamma_{k+1}) = 1/(2L)`, the same coefficient as the descent lemma. With that choice the point

`x_{k+1}=y_k-(1/L)grad f(y_k)`

satisfies `f(x_{k+1}) <= f(y_k) - (1/(2L))||grad f(y_k)||^2`, so the gradient-norm part is paid for term by term.

The second dangerous term is worse because it has no sign. The inner product `(1-alpha_k)<grad f(y_k), ...>` can be positive or negative depending on the direction of the gradient, and I have no lower bound for it. The only safe thing is to make the vector multiplying the gradient vanish:

`(alpha_k gamma_k/gamma_{k+1})(v_k-y_k)+x_k-y_k=0`.

Solving for `y_k` (collecting the `y_k` terms, the coefficient is `1+alpha_k gamma_k/gamma_{k+1}=(gamma_k+alpha_k mu)/gamma_{k+1}` after using the `gamma` recurrence) gives

`y_k=(alpha_k gamma_k v_k+gamma_{k+1}x_k)/(gamma_k+alpha_k mu)`.

So the gradient is not read at the current iterate and then decorated with momentum. The bookkeeping forces a look-ahead point, a blend of the current output point `x_k` and the minimizer `v_k` of the accumulated lower model, and the gradient is read there. Momentum is what is left after I eliminate the hidden model variables `v_k,gamma_k`; its coefficient is whatever cancels the cross term, not an inertial guess.

With both choices in place the chain reads `f(x_{k+1}) <= f(y_k)-(1/(2L))||grad f(y_k)||^2` and, from the recurrence, `phi_{k+1}^* >= f(y_k)-(1/(2L))||grad f(y_k)||^2` once the cross term is killed, so `f(x_{k+1}) <= phi_{k+1}^*`. The induction closes only because the two pinned quantities are the *same* `1/(2L)`; I want to make sure I have not fooled myself about that, so before reading off the rate I will trace the resulting scheme and check the per-step descent lemma actually holds on a concrete function.

Eliminating the hidden variables in the merely convex case `mu=0` leaves the two-sequence form `x_{k+1}=y_k-(1/L)grad f(y_k)`, `y_{k+1}=x_{k+1}+((a_k-1)/a_{k+1})(x_{k+1}-x_k)` with `a_{k+1}=(1+sqrt(1+4a_k^2))/2`. Running this on the pseudo-Huber function `f(x)=sqrt(1+||x||^2)-1` (convex, gradient-Lipschitz with `L=1`, minimum `0`), which is deliberately not a quadratic, from `x_0=(3,-4,1,2)` I get the per-step check `f(x_{k+1}) <= f(y_k)-(1/2)||grad f(y_k)||^2`: over 399 iterations the right side never undershoots the left, zero violations. That is the descent lemma the matching `1/(2L)` was supposed to deliver, and it survives on a non-quadratic, which is exactly what the spectral proof could not promise. The gap `f(x_k)-f*` against `C/(k+2)^2` with `C=2L||x_0-x*||^2=60` stays below the bound throughout: at `k=1,2` the ratio `gap*(k+2)^2/C` is `0.54` and `0.71`, and by `k=5` the iterate is already at the minimizer. So the `1/k^2` envelope holds and is not loose nonsense.

The rate constant follows from the `a_k` recurrence. I check the telescoping identity it is supposed to satisfy, `a_{k+1}^2-a_{k+1}=a_k^2`: starting from `a_0=1`, `a_1=1.618`, `a_2=2.194`, the left side `a_1^2-a_1=1.000` equals `a_0^2=1`, then `a_2^2-a_2=2.618` equals `a_1^2=2.618`, and continuing, `a_{k+1}^2-a_{k+1}` against `a_k^2` reads `1.0, 2.618, 4.812, 7.561, ...` matching `a_k^2` to all digits. Iterating far out, `a_{2000}=1003.2`, i.e. `a_k = k/2 + O(1)`, so `lambda_k ~ 1/a_k^2 ~ 4/k^2`. That is the accelerated smooth-convex rate, and it matches the tridiagonal lower-bound order.

In the strongly convex case there is a fixed point. If I start with `gamma_0=mu`, then `gamma_{k+1}=(1-alpha_k)gamma_k+alpha_k mu=mu` for all `k`, and the equation `L alpha^2=gamma=mu` gives `alpha=sqrt(mu/L)=1/sqrt(kappa)`, constant. The two-sequence coefficient becomes

`(1-alpha)/(1+alpha) = (sqrt(kappa)-1)/(sqrt(kappa)+1)`,

and `lambda_k=(1-alpha)^k`, so the certificate gap has exponent `1/sqrt(kappa)`. To see whether the resulting `f(x_k)-f* <= ((L+mu)/2)||x_0-x*||^2 exp(-k sqrt(mu/L))` is a true envelope and not a hope, I run the fixed-momentum scheme on the diagonal quadratic with `mu=1`, `L=4` (so `kappa=4`, `theta=1/3`) from `x_0=(10,-10)`. The prefactor is `((L+mu)/2)R^2=500`. The actual gaps `28.1, 12.5, 4.88, ...` sit under the bound `303, 184, 112, ...` with ratio starting at `0.093` and falling monotonically to `~10^{-5}` by `k=15`. The bound holds with margin, as it should for a worst-case guarantee that is tight only on the hard instance. This is the same square-root scale that showed up in the quadratic methods and in the lower bound, but here it is certified by accumulated lower models, not by diagonalizing a Hessian.

I also wanted concrete reassurance that letting the value rise is not just a theoretical loophole. On an ill-conditioned quadratic (`L=100`, the other coordinate flat) the smooth-convex scheme's value `f(x_k)` does increase across a run of iterations, `38` through `49` in my trace, before resuming its fall. So the method is genuinely non-monotone on real inputs, and the only thing that decreases monotonically is `lambda_k` and with it the certificate `phi_k^*-f`. That is the whole point of abandoning per-step descent: I traded a quantity that must drop every step for one that drops at the lower-bound rate.

The 1983 potential proof is the same accounting in a more compressed form. It defines `p_k=(a_k-1)(x_{k-1}-x_k)` and tracks `a_k^2(f(x_k)-f*)+||p_k-x_k+x*||^2`. The line-search inequality, convexity, and exactly the identity `a_{k+1}^2-a_{k+1}=a_k^2` I checked above make that quantity telescope, landing on `C/(k+2)^2`; the later estimate-sequence view explains why the extrapolation coefficient is the only one that keeps the global certificate intact rather than being a tuned constant.

So the method is a coupling between a point sequence and a hidden lower-bound sequence. The extrapolated point is where the next lower model must be sampled so that the model minimum stays above the next function value, and the momentum coefficient is the visible trace of that coupling. The traces above are why I believe it rather than merely assert it: the descent lemma held at every step on a non-quadratic, the `1/k^2` and `exp(-k/sqrt(kappa))` envelopes held with margin, the `a_k` identity is exact, and the value really does rise on some iterations while the certificate keeps shrinking.
