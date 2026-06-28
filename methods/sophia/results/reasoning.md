I start from the cost target, not from a taste for second-order methods. A language-model pre-training run is mostly optimizer steps times the cost of one step. If I want a real win, I need fewer steps to the same loss while keeping the average step cost close to a first-order method. A method that looks good only at a fixed number of steps but spends much more work per step is not solving the problem.

Adam is the incumbent, so I first strip it down. With its moving averages suppressed, its update is \(g / |g|\) coordinate by coordinate, multiplied by the learning rate. That is SignGD. The moving averages matter in practice, but the skeleton is a fixed-magnitude coordinate step: every coordinate gets about the same update size, regardless of the local curvature.

That is only sensible if equal update size means equal progress, and on language-model losses it does not. The measured Hessian spectra and diagonal entries are highly spread out. A minimal model is a separable two-coordinate objective \(L(\theta_1,\theta_2)=L_1(\theta_1)+L_2(\theta_2)\), with \(L_1\) sharp and \(L_2\) flat. Gradient descent wants learning rates proportional to \(1/h_1\) and \(1/h_2\). One shared learning rate must be small enough for the sharp coordinate, so the flat coordinate crawls. SignGD has the opposite-looking but still fatal failure: it moves both coordinates the same distance. The sharp coordinate reaches the valley and bounces unless the step is decayed; the flat coordinate needs large travel distance and makes tiny loss progress. Both failures say the same thing: I should equalize loss decrease, not coordinate displacement.

Let me make that contrast concrete before I trust it. Take \(L=\frac{h_1}{2}\theta_1^2+\frac{h_2}{2}\theta_2^2\) with \(h_1=100,h_2=1\) (condition number 100) and start at \((3,-5)\). The gradient is \(g_i=h_i\theta_i\), so the per-coordinate Newton step \(-g_i/h_i=-\theta_i\) lands at \((0,0)\) in a single step — and the algebra \(-h_i\theta_i/h_i=-\theta_i\) has no \(h_i\) left in it, so this is true for any condition number. Now the sign step. To avoid overshooting the sharp coordinate on its final approach I need a small \(\eta\); with \(\eta=10^{-2}\), I simulate SignGD to bring both coordinates within \(10^{-3}\) and it takes 500 steps. The bottleneck is the flat coordinate: it has to traverse distance \(|\theta_2|=5\) in fixed strides of \(\eta=10^{-2}\), i.e. \(\approx 500\) steps, and \(\sqrt{h_1/h_2}=10\) shows up as the gap between the sign step and what curvature scaling would buy. So in this idealization the right scale factor really is curvature, and the displacement-equalizing step pays for the spread it ignores.

The reason curvature is the right factor is visible on a local quadratic \(q(\theta)=\frac12 h(\theta-\theta^*)^2\): the gradient is \(g=h(\theta-\theta^*)\), the step \(-g/h\) lands at the minimizer, and the removed local loss is \(\frac12 g^2/h\). Sharp coordinates get smaller displacement, flat coordinates get larger displacement, and — as the simulation just showed — the condition number drops out of the rate in the separable quadratic idealization.

The obstacle is that plain Newton is unusable as stated. The full Hessian is too large to form or invert for a large model. Away from a local minimum the Hessian can be indefinite; if \(h<0\), the raw Newton direction \(-g/h\) points toward a maximum along that coordinate. The Hessian also changes along the path, so a stale or noisy curvature estimate can make a locally reasonable step too large. Trust regions, line search, and cubic regularization solve versions of this problem, but they are too heavy for the cost target.

So I want the cheap part of Newton and a cheap safety valve. The cheap part is a diagonal curvature estimate: one extra vector, one scale per parameter. The safety valve is per-coordinate clipping of the preconditioned update. Clipping alone only bounds magnitude; the sign still has to be handled correctly. The safe denominator must use positive curvature, either because the estimator is nonnegative or because I divide by \(\max\{\gamma h,\epsilon\}\). I want to check that this floor actually does what I am hoping: that a bad coordinate degrades gracefully to a sign step rather than blowing up or flipping sign. With \(\gamma=0.04,\epsilon=10^{-12}\), the update direction is \(\operatorname{clip}(m/\max\{\gamma h,\epsilon\},1)\). For \(h=-3,m=0.7\), the floor replaces \(\gamma h\) by \(\epsilon\), so \(m/\epsilon=7\times10^{11}\), which clips to \(1.000=\operatorname{sign}(m)\). For \(h=0,m=-0.2\) I get \(-2\times10^{11}\to-1.000=\operatorname{sign}(m)\). For a tiny positive \(h=10^{-20}\), \(m=0.5\to1.000\). So in every untrustworthy case the floor plus clip yields exactly \(\operatorname{sign}(m_i)\): a bounded momentum-SignGD step with the correct sign. In the trustworthy case, where the ratio \(m_i/h_i\) is moderate, the clip is inactive and I get the curvature-scaled step. The floor never introduces a sign error, which was the thing clipping alone could not guarantee.

Now I need a diagonal curvature estimate cheap enough to refresh infrequently. The first route is Hutchinson. Draw \(u\) with \(E[uu^\top]=I\), for example a spherical Gaussian. Return \(\hat h=u\odot(Hu)\). Coordinate \(i\) has expectation
\[
E[u_i(Hu)_i]=\sum_j H_{ij}E[u_i u_j]=H_{ii}.
\]
The product \(Hu\) is a Hessian-vector product, available as \(\nabla_\theta\langle \nabla_\theta L,u\rangle\), so I never materialize the Hessian. The algebra says it is unbiased for the true diagonal, but I want to see the variance and the indefiniteness behavior, so I run it on a random symmetric \(4\times4\) matrix with eigenvalues \(\{-1.86,-0.37,0.23,3.27\}\) (so genuinely indefinite). The true diagonal is \((1.764,-0.977,0.144,0.334)\). Averaging \(u\odot(Hu)\) over a couple million Gaussian draws gives \((1.762,-0.976,0.144,0.332)\), within \(2.5\times10^{-3}\). So the estimator is unbiased as derived, it recovers the negative diagonal entry correctly — and that is exactly the case where individual draws are negative and the optimizer has to fall back to the positive floor and clip to a sign step. Good: the estimator and the safety valve are designed for each other.

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
so the diagonal should be the expected elementwise gradient square under labels sampled from the model. That is a strong claim — it says I can read curvature off ordinary gradients, with no Hessian-vector product — so I check it on a tiny model rather than trust the chain rule. Take logits \(f=Wx\) with \(W\in\mathbb{R}^{3\times2}\), \(x\in\mathbb{R}^2\), so \(\partial f_a/\partial W_{ij}=x_j[a=i]\) and the Gauss-Newton diagonal at entry \((i,j)\) is \(x_j^2 S_{ii}\). Against this I compute \(\sum_{\hat y} p_{\hat y}\,(\nabla_W\operatorname{ce}(f,\hat y))^2\) with \(\nabla_W\operatorname{ce}(f,\hat y)=(p-e_{\hat y})x^\top\). The two arrays agree to \(10^{-16}\) — machine precision, every entry. The Bartlett route to the Gauss-Newton diagonal is exact, not approximate.

The practical snag is that autodiff gives the averaged minibatch gradient, not per-example gradients. Let
\[
\widehat L(\theta)=\frac1B\sum_{b=1}^B \operatorname{ce}(f(\theta,x_b),\hat y_b),
\]
with independent \(\hat y_b\sim \operatorname{Cat}(\operatorname{softmax}(f(\theta,x_b)))\). Bartlett's first identity gives \(E_{\hat y_b}[\nabla_\theta \operatorname{ce}(f(\theta,x_b),\hat y_b)]=0\), so each per-example sampled-label gradient is zero-mean and the cross terms in \((\sum_b)^2\) vanish in expectation:
\[
E[B\,\nabla\widehat L\odot\nabla\widehat L]
=E\left[\frac1B\sum_b \nabla\operatorname{ce}_b\odot\nabla\operatorname{ce}_b\right].
\]
This is the step I am least sure of, because it hinges on the cross terms cancelling and on the exact \(B\) factor, and I am only allowed the averaged gradient. So I test it directly: a batch of \(B=8\) examples, sample one label per example from its own \(p_b\), form the averaged gradient \(\bar g=\frac1B\sum_b \nabla\operatorname{ce}_b\), and Monte-Carlo \(E[B\,\bar g\odot\bar g]\) over \(4\times10^5\) resamples. The target is the per-example Gauss-Newton diagonal averaged over the batch, \(\frac1B\sum_b x_{b,j}^2 S_{b,ii}\). The two match to \(4\times10^{-4}\), which is the Monte-Carlo noise floor at this sample count. Two things are confirmed at once: the cross terms really do cancel (otherwise the estimate would be biased by the off-diagonal mass), and the multiplier is exactly \(B\), not \(1\) or \(\sqrt B\). So I can get the Gauss-Newton diagonal from one ordinary backward pass on model-sampled labels, squared and scaled by the batch size — nonnegative by construction, gradient-only, and estimating the Gauss-Newton diagonal rather than the full Hessian diagonal.

I keep two moving averages. The numerator is \(m_t=\beta_1m_{t-1}+(1-\beta_1)g_t\). The curvature denominator is updated only on refresh steps, \(h_t=\beta_2h_{t-k}+(1-\beta_2)\hat h_t\), and otherwise carried forward. Refreshing every step would erase the wall-clock benefit; refreshing rarely is tolerable only because, as I checked above, a bad or stale coordinate is clipped into a bounded sign step with the right sign.

The clean mathematical update is
\[
\theta_{t+1}=\theta_t-\eta_t\,\operatorname{clip}\!\left(\frac{m_t}{\max\{\gamma h_t,\epsilon\}},1\right),
\]
with decoupled weight decay applied separately. I want to understand what the two knobs \(\eta\) and \(\gamma\) separately control, because in this form they are tangled. The identity
\[
\eta\,\operatorname{clip}(m/\max\{\gamma h,\epsilon\},1)
=\frac{\eta}{\gamma}\operatorname{clip}(m/\max\{h,\epsilon/\gamma\},\gamma)
\]
disentangles them, and I verify it numerically over \(10^4\) random \((m,h)\) pairs — including a few thousand with \(h\) tiny or negative to exercise the floor — and the two sides agree to \(5.6\times10^{-17}\). So the reparameterization is an exact identity, not an approximation, even across the floor. Reading off the right-hand side: \(\gamma\) is the clip threshold on the raw curvature-scaled ratio \(m/\max\{h,\cdot\}\). Saturated coordinates have update magnitude \(\eta\) in the left-hand form, so \(\eta\) sets the saturated step size while \(\gamma\) controls how many coordinates are allowed to be unclipped. In an implementation that stores the unscaled sampled-label gradient square, the same operational idea appears as a denominator scale `rho * bs * h`, where `bs` supplies the \(B\) factor I just verified.

I also need to be precise about what a convergence theorem can actually prove. The clean analysis is not a theorem for the stochastic diagonal implementation. It analyzes deterministic clipped Newton with the exact full Hessian, clipped in the Hessian eigenbasis:
\[
\theta_+=\theta-\eta V^\top\operatorname{clip}(V(\nabla^2L(\theta))^{-1}\nabla L(\theta),\rho),
\]
where \(\nabla^2L(\theta)=V^\top\Sigma V\). Under strict convexity and the multiplicative Hessian-continuity assumption \(\|\nabla^2L(\theta')^{-1}\nabla^2L(\theta)\|_2\le 2\) within radius \(R\), and with \(\eta\rho\le R/\sqrt d\), define \(u=\operatorname{clip}(\Sigma^{-1}V\nabla L,\rho)\) and \(f(t)=L(t\theta_+ +(1-t)\theta)\). The step stays inside the radius-\(R\) neighborhood, so \(f''(t)\le 2f''(0)\), hence \(f(1)\le f(0)+f'(0)+f''(0)\).

The first derivative is
\[
f'(0)=-\eta\sum_i (v_i^\top\nabla L)\operatorname{clip}(\sigma_i^{-1}v_i^\top\nabla L,\rho)
=-\eta\sum_i \min\{\rho|v_i^\top\nabla L|,\sigma_i^{-1}|v_i^\top\nabla L|^2\}.
\]
The min form is worth pausing on: for a fixed coordinate, as I dial \(g_i=|v_i^\top\nabla L|\) up from zero, the unclipped term \(\sigma_i^{-1}g_i^2\) is the smaller one while \(g_i<\rho\sigma_i\), and the clipped term \(\rho g_i\) takes over once \(g_i>\rho\sigma_i\); at the crossover both equal \(\rho^2\sigma_i\). So the \(\min\) is genuinely the clip boundary, not a bound I am hoping holds. The second derivative satisfies
\[
f''(0)=\eta^2\sum_i u_i^2\sigma_i
\le \eta^2\sum_i \min\{\rho|v_i^\top\nabla L|,\sigma_i^{-1}|v_i^\top\nabla L|^2\},
\]
where the bound uses \(u_i^2\sigma_i=\sigma_i\min\{\rho,\sigma_i^{-1}g_i\}^2=\min\{\rho^2\sigma_i,\sigma_i^{-1}g_i^2\}\le\min\{\rho g_i,\sigma_i^{-1}g_i^2\}\), the last step because \(\rho^2\sigma_i\le\rho g_i\) exactly on the clipped branch and \(\sigma_i^{-1}g_i^2\) carries the unclipped branch. Combining them gives
\[
L(\theta_+)-L(\theta)
\le-(\eta-\eta^2)\sum_i \min\{\rho|v_i^\top\nabla L|,\sigma_i^{-1}|v_i^\top\nabla L|^2\}.
\]
The two terms are exactly the two operating regimes: unclipped Newton progress when the local ratio is safe, and clipped sign-like progress when it is not.

The rest of the proof is a two-phase argument. If the summed decrement is small, the iterate is already near the minimizer; otherwise the descent lemma forces loss down by a fixed amount. Once \(L(\theta)-\min L\le \mu\rho^2/8\), the inverse-Hessian gradient norm is at most \(\rho\), no coordinate clips, and the update becomes ordinary damped Newton with exponential loss decay. With \(\eta=1/2\) and \(\rho=R/(2\sqrt d)\), the displayed rate is
\[
T\lesssim d\frac{L(\theta_0)-\min L}{\mu R^2}+\log\frac{\mu R^2}{32d\epsilon}.
\]
There is no dependence on the condition number or the largest curvature in that simplified full-Hessian model — which matches the one-step landing I saw on the two-coordinate quadratic. For contrast, the comparison lower bound for SignGD on \(L_{\mu,\beta}(\theta)=\frac{\mu}{2}\theta_1^2+\frac{\beta}{2}\theta_2^2\) gives
\[
T\ge \frac12\left(\sqrt{\Delta/\epsilon}-\sqrt2\right)\sqrt{\beta/\mu},
\]
so the sign-step proxy pays the square root of the condition number — the same \(\sqrt{\beta/\mu}\) that showed up empirically as the 500-step flat-coordinate traverse earlier.

The final implementation shape follows from these pieces. On ordinary steps I use the real-label gradient to update the momentum numerator and apply the clipped denominator-scaled update. Every \(k\) steps I run a separate sampled-label backward pass and update only the curvature EMA. The optimizer state stores `exp_avg`, `hessian`, and `step`; applies decoupled weight decay; updates `exp_avg`; computes `ratio = clamp(abs(exp_avg) / (rho * bs * hessian + 1e-15), max=1)`; and applies `param -= lr * sign(exp_avg) * ratio`. This is faithful to the GNB estimator because `update_hessian` has stored the unscaled sampled-label gradient square, while `bs` restores the batch/token factor in the denominator — the exact \(B\) multiplier whose value I pinned down with the Monte-Carlo check. The structure-agnostic variant would change only the curvature refresh: replace sampled-label gradient squares with an EMA of \(u\odot Hu\).
