I start from the cost target, not from a taste for second-order methods. A language-model pre-training run is mostly optimizer steps times the cost of one step. If I want a real win, I need fewer steps to the same loss while keeping the average step cost close to a first-order method. A method that looks good only at a fixed number of steps but spends much more work per step is not solving the problem.

Adam is the incumbent, so I first strip it down. With its moving averages suppressed, its update is \(g / |g|\) coordinate by coordinate, multiplied by the learning rate. That is SignGD. The moving averages matter in practice, but the skeleton is a fixed-magnitude coordinate step: every coordinate gets about the same update size, regardless of the local curvature.

That is only sensible if equal update size means equal progress, and on language-model losses it does not. The measured Hessian spectra and diagonal entries are highly spread out. A minimal model is a separable two-coordinate objective \(L(\theta_1,\theta_2)=L_1(\theta_1)+L_2(\theta_2)\), with \(L_1\) sharp and \(L_2\) flat. Gradient descent wants learning rates proportional to \(1/h_1\) and \(1/h_2\). One shared learning rate must be small enough for the sharp coordinate, so the flat coordinate crawls. SignGD has the opposite-looking but still fatal failure: it moves both coordinates the same distance. The sharp coordinate reaches the valley and bounces unless the step is decayed; the flat coordinate needs large travel distance and makes tiny loss progress. Both failures say the same thing: I should equalize loss decrease, not coordinate displacement.

On a local quadratic \(q(\theta)=\frac12 h(\theta-\theta^*)^2\), the gradient is \(g=h(\theta-\theta^*)\), and the step \(-g/h\) lands at the minimizer. The removed local loss is \(\frac12 g^2/h\). Curvature is therefore exactly the scale factor I need: sharp coordinates get smaller displacement, flat coordinates get larger displacement, and the condition number stops dictating the rate in the separable quadratic idealization.

The obstacle is that plain Newton is unusable as stated. The full Hessian is too large to form or invert for a large model. Away from a local minimum the Hessian can be indefinite; if \(h<0\), the raw Newton direction \(-g/h\) points toward a maximum along that coordinate. The Hessian also changes along the path, so a stale or noisy curvature estimate can make a locally reasonable step too large. Trust regions, line search, and cubic regularization solve versions of this problem, but they are too heavy for the cost target.

So I want the cheap part of Newton and a cheap safety valve. The cheap part is a diagonal curvature estimate: one extra vector, one scale per parameter. The safety valve is per-coordinate clipping of the preconditioned update. Clipping alone only bounds magnitude; the sign still has to be handled correctly. The safe denominator must use positive curvature, either because the estimator is nonnegative or because I divide by \(\max\{\gamma h,\epsilon\}\). Then a negative or tiny \(h_i\) is replaced by \(\epsilon\), so \(m_i/\epsilon\) has the sign of the momentum gradient and clips to \(\operatorname{sign}(m_i)\). In the untrustworthy case I get a bounded momentum-SignGD step. In the trustworthy case, where the ratio \(m_i/h_i\) is moderate, the clip is inactive and I get the curvature-scaled step.

Now I need a diagonal curvature estimate cheap enough to refresh infrequently. The first route is Hutchinson. Draw \(u\) with \(E[uu^\top]=I\), for example a spherical Gaussian. Return \(\hat h=u\odot(Hu)\). Coordinate \(i\) has expectation
\[
E[u_i(Hu)_i]=\sum_j H_{ij}E[u_i u_j]=H_{ii}.
\]
The product \(Hu\) is a Hessian-vector product, available as \(\nabla_\theta\langle \nabla_\theta L,u\rangle\), so I never materialize the Hessian. This estimator is unbiased for the true diagonal Hessian, but individual coordinates may be negative; those cases rely on the positive floor and clip to fall back to SignGD.

The second route uses the loss structure. For cross-entropy \(\ell(\theta,(x,y))=\operatorname{ce}(f(\theta,x),y)\), with logits \(f\in\mathbb{R}^V\), the chain rule gives
\[
\nabla_\theta^2 \ell = J_\theta f\, S\, J_\theta f^\top + J_{\theta\theta}f[q],
\]
where \(S=\partial_t^2\operatorname{ce}(t,y)|_{t=f(\theta,x)}\) and \(q=\partial_t\operatorname{ce}(t,y)|_{t=f(\theta,x)}\). The first term is the Gauss-Newton matrix. Dropping the second term is a bias, but it gives a positive semidefinite curvature surrogate because cross-entropy is convex in logits, so \(S\succeq 0\).

For softmax cross-entropy, \(S=\operatorname{diag}(p)-pp^\top\), with \(p=\operatorname{softmax}(f)\), and this does not depend on the realized label. If I sample \(\hat y\sim \operatorname{Cat}(p)\), cross-entropy against \(\hat y\) is the categorical negative log-likelihood. Bartlett's second identity gives
\[
S=E_{\hat y}\left[(\partial_t\operatorname{ce}(f,\hat y))(\partial_t\operatorname{ce}(f,\hat y))^\top\right].
\]
Multiplying by the logit Jacobian gives
\[
J_\theta f\,S\,J_\theta f^\top
=E_{\hat y}\left[\nabla_\theta \operatorname{ce}(f,\hat y)\nabla_\theta \operatorname{ce}(f,\hat y)^\top\right],
\]
so the diagonal is the expected elementwise gradient square under labels sampled from the model.

The practical snag is that autodiff gives the averaged minibatch gradient, not per-example gradients. Let
\[
\widehat L(\theta)=\frac1B\sum_{b=1}^B \operatorname{ce}(f(\theta,x_b),\hat y_b),
\]
with independent \(\hat y_b\sim \operatorname{Cat}(\operatorname{softmax}(f(\theta,x_b)))\). Bartlett's first identity gives \(E_{\hat y_b}[\nabla_\theta \operatorname{ce}(f(\theta,x_b),\hat y_b)]=0\), so the cross terms vanish:
\[
E[B\,\nabla\widehat L\odot\nabla\widehat L]
=E\left[\frac1B\sum_b \nabla\operatorname{ce}_b\odot\nabla\operatorname{ce}_b\right].
\]
That is the diagonal of the minibatch Gauss-Newton matrix. This gives the Gauss-Newton-Bartlett estimator: sample labels from the model, run one ordinary backward pass, square the averaged gradient, and multiply by the batch size \(B\). It is nonnegative and gradient-only, but it estimates the Gauss-Newton diagonal rather than the full Hessian diagonal.

I keep two moving averages. The numerator is \(m_t=\beta_1m_{t-1}+(1-\beta_1)g_t\). The curvature denominator is updated only on refresh steps, \(h_t=\beta_2h_{t-k}+(1-\beta_2)\hat h_t\), and otherwise carried forward. Refreshing every step would erase the wall-clock benefit; refreshing rarely is tolerable only because a bad or stale coordinate is clipped into a bounded sign step.

The clean mathematical update is
\[
\theta_{t+1}=\theta_t-\eta_t\,\operatorname{clip}\!\left(\frac{m_t}{\max\{\gamma h_t,\epsilon\}},1\right),
\]
with decoupled weight decay applied separately. The identity
\[
\eta\,\operatorname{clip}(m/\max\{\gamma h,\epsilon\},1)
=\frac{\eta}{\gamma}\operatorname{clip}(m/\max\{h,\epsilon/\gamma\},\gamma)
\]
shows the purpose of \(\gamma\): it is the clip threshold on the raw curvature-scaled ratio after reparameterization. Saturated coordinates have update magnitude \(\eta\) in the left-hand form, so \(\eta\) sets the saturated step size while \(\gamma\) controls how many coordinates are allowed to be unclipped. In an implementation that stores the unscaled sampled-label gradient square, the same operational idea appears as a denominator scale `rho * bs * h`, where `bs` supplies the \(B\) factor.

I also need to be precise about what the convergence theorem actually proves. The clean analysis is not a theorem for the stochastic diagonal implementation. It analyzes deterministic clipped Newton with the exact full Hessian, clipped in the Hessian eigenbasis:
\[
\theta_+=\theta-\eta V^\top\operatorname{clip}(V(\nabla^2L(\theta))^{-1}\nabla L(\theta),\rho),
\]
where \(\nabla^2L(\theta)=V^\top\Sigma V\). Under strict convexity and the multiplicative Hessian-continuity assumption \(\|\nabla^2L(\theta')^{-1}\nabla^2L(\theta)\|_2\le 2\) within radius \(R\), and with \(\eta\rho\le R/\sqrt d\), define \(u=\operatorname{clip}(\Sigma^{-1}V\nabla L,\rho)\) and \(f(t)=L(t\theta_+ +(1-t)\theta)\). The step stays inside the radius-\(R\) neighborhood, so \(f''(t)\le 2f''(0)\), hence \(f(1)\le f(0)+f'(0)+f''(0)\).

The first derivative is
\[
f'(0)=-\eta\sum_i (v_i^\top\nabla L)\operatorname{clip}(\sigma_i^{-1}v_i^\top\nabla L,\rho)
=-\eta\sum_i \min\{\rho|v_i^\top\nabla L|,\sigma_i^{-1}|v_i^\top\nabla L|^2\}.
\]
The second derivative satisfies
\[
f''(0)=\eta^2\sum_i u_i^2\sigma_i
\le \eta^2\sum_i \min\{\rho|v_i^\top\nabla L|,\sigma_i^{-1}|v_i^\top\nabla L|^2\}.
\]
Combining them gives
\[
L(\theta_+)-L(\theta)
\le-(\eta-\eta^2)\sum_i \min\{\rho|v_i^\top\nabla L|,\sigma_i^{-1}|v_i^\top\nabla L|^2\}.
\]
The two terms are exactly the intended cases: unclipped Newton progress when the local ratio is safe, and clipped sign-like progress when it is not.

The rest of the proof is a two-phase argument. If the summed decrement is small, the iterate is already near the minimizer; otherwise the descent lemma forces loss down by a fixed amount. Once \(L(\theta)-\min L\le \mu\rho^2/8\), the inverse-Hessian gradient norm is at most \(\rho\), no coordinate clips, and the update becomes ordinary damped Newton with exponential loss decay. With \(\eta=1/2\) and \(\rho=R/(2\sqrt d)\), the displayed rate is
\[
T\lesssim d\frac{L(\theta_0)-\min L}{\mu R^2}+\log\frac{\mu R^2}{32d\epsilon}.
\]
There is no dependence on the condition number or the largest curvature in that simplified full-Hessian model. The comparison lower bound for SignGD on \(L_{\mu,\beta}(\theta)=\frac{\mu}{2}\theta_1^2+\frac{\beta}{2}\theta_2^2\) gives
\[
T\ge \frac12\left(\sqrt{\Delta/\epsilon}-\sqrt2\right)\sqrt{\beta/\mu},
\]
so the sign-step proxy pays the square root of the condition number.

The final implementation shape follows from these pieces. On ordinary steps I use the real-label gradient to update the momentum numerator and apply the clipped denominator-scaled update. Every \(k\) steps I run a separate sampled-label backward pass and update only the curvature EMA. The optimizer state stores `exp_avg`, `hessian`, and `step`; applies decoupled weight decay; updates `exp_avg`; computes `ratio = clamp(abs(exp_avg) / (rho * bs * hessian + 1e-15), max=1)`; and applies `param -= lr * sign(exp_avg) * ratio`. This is faithful to the GNB estimator because `update_hessian` has stored the unscaled sampled-label gradient square, while `bs` restores the batch/token factor in the denominator. The structure-agnostic variant would change only the curvature refresh: replace sampled-label gradient squares with an EMA of \(u\odot Hu\).
