We need fast guided sampling of a pre-trained diffusion model in the extreme few-step regime — five to ten network evaluations — with images that stay sharp and prompt-faithful under large classifier-free or classifier guidance, and we need it training-free, working with weights as they are. The fast solvers I already have get me most of the way: the data-prediction multistep exponential-integrator family (DPM-Solver, DPM-Solver++, DEIS) cuts unconditional sampling to roughly ten to twenty steps, and DPM-Solver++ in particular restored stability under heavy guidance by moving to the data-prediction parameterization. But push the budget down to five or ten steps and the wheels come off: the half-log-SNR interval $h = \lambda_t - \lambda_s$ per step becomes large, the $O(h^k)$ truncation term is no longer small, and every one of these solvers leaves a visible per-step error. The reflex is to raise the order, but raising the order of a *predictor* solver buys less and less, and under guidance — where the model's derivatives are amplified — it can actively hurt. So the right question is not "how do I get more order" but "what kind of error am I leaving uncorrected, and is there a cheaper lever than order."

The structural fact is that every solver I use is *predictor-only*: at each step it forms an estimate of $x_t$ from information it already has and then commits to it and marches on, with the leading truncation error of that step baked in and never refined. Classical ODE numerics has exactly the remedy for this — a predictor-corrector pair. An Adams–Bashforth predictor takes the step, you evaluate the right-hand side *at the predicted point*, and an Adams–Moulton corrector uses that fresh evaluation to refine the step and pick up one extra order. The corrector is the cheapest order one can buy, because it needs only a single new function evaluation at the predicted point. In a generic ODE that evaluation is real cost. But in a *multistep diffusion* loop, look at what happens: at the start of step $i+1$, the predictor's base evaluation is the network applied at exactly the point step $i$ just predicted — which is precisely the evaluation a corrector for step $i$ would require. So the corrector is *free*. I reuse the next step's predictor evaluation to correct the previous step, and the network-call count never changes. That is the whole opening: a predictor-corrector that costs the same NFE as the predictor alone, but with each step's leading truncation error corrected.

The reason this is not already standard is derivation cost. A corrector of order $p$ is a different formula from the predictor of order $p$, and from the corrector of order $p+1$, and each is a separate hand-derivation of exponential-integrator coefficients. People derive a second-order and a third-order solver and stop, because each new order is bespoke algebra. So the real obstacle is not the corrector idea — it is the absence of a *unified* form that yields predictor and corrector at arbitrary order from one template. If I can write both as the same update, parameterized by order, then "add a corrector" and "go one order higher" collapse into the same one-line change, and the corrector becomes available for free at whatever order I happen to be running.

I propose UniPC: a unified predictor-corrector framework with a multistep predictor (UniP) and a corrector (UniC) sharing one analytical update of arbitrary order, built on the data-prediction diffusion ODE. The starting point is the exact data-prediction step between the current noise level $s_0$ and the next level $t$,
$$x_t = \frac{\sigma_t}{\sigma_{s_0}}\,x_{s_0} + \sigma_t \int_{\lambda_{s_0}}^{\lambda_t} e^{\lambda}\, x_\theta(\lambda)\, d\lambda,$$
where $\lambda_t = \log(\alpha_t/\sigma_t)$ is the strictly monotone half-log-SNR. Everything high-order lives in approximating that integral. I expand $x_\theta(\lambda)$ in a Taylor series about the left endpoint $\lambda_{s_0}$. Holding $x_\theta$ constant at its current value $m_0 = x_\theta(x_{s_0}, s_0)$ integrates to the first-order base step: with $h = \lambda_t - \lambda_{s_0}$ and the identity $\sigma_t e^{\lambda_t} = \alpha_t$, the base term is
$$x_t^{(\text{base})} = \frac{\sigma_t}{\sigma_{s_0}}\,x_{s_0} - \alpha_t\, h_{\varphi_1}\, m_0, \qquad h_{\varphi_1} = e^{hh} - 1,$$
where $hh = -h$ in the data-prediction convention (the `predict_x0` flag flips the sign of $h$). The higher-order terms involve the derivatives $x_\theta^{(n)}$ at $\lambda_{s_0}$, each weighted by an integral of $e^{\lambda}(\lambda-\lambda_{s_0})^n$ — that is, by a $\varphi$ function evaluated at $hh$, with the recurrence $\varphi_{k+1}(z) = (\varphi_k(z) - 1/k!)/z$.

I don't have those derivatives in closed form, so I estimate them from finite differences of past model outputs. Keeping the last few data predictions $m_0, m_1, m_2, \dots$ at past noise levels with half-log-SNRs $\lambda_{s_0}, \lambda_{s_1}, \dots$, I define the ratios $r_k = (\lambda_{s_k} - \lambda_{s_0})/h$ (so $r_0 = 0$ and the past points have $r_k \neq 0$) and the scaled differences $D1_k = (m_k - m_0)/r_k$. A linear combination $\sum_k \rho_k D1_k$ of these, with the right coefficients $\rho_k$, reproduces the Taylor correction to whatever order I have points for. The right coefficients come from matching the method's update to the exact integral's Taylor expansion term by term — which is a *linear system*. Collect the powers of the ratios into a matrix $R$ whose $i$-th row is $r_k^{\,i-1}$ (Vandermonde-like in the ratios), and a right-hand side $b$ whose $i$-th entry is the $\varphi$-derived coefficient the exact integral assigns to the $i$-th Taylor term, divided by a chosen scalar $B(h)$. Then
$$R\,\rho = b$$
gives exactly the $\rho_k$ that make the finite-difference combination match the exact correction to the available order. That single linear solve produces the high-order coefficients at *any* order, with no per-order hand derivation — the order is simply how many rows and points I include. Concretely $b$ is built iteratively alongside $R$: starting from $h_{\varphi_1} = e^{hh}-1$, set $h_{\varphi_k} = h_{\varphi_1}/hh - 1$, and for $i = 1,\dots,\text{order}$ append the row $r_k^{\,i-1}$ to $R$ and the entry $h_{\varphi_k}\, i!/B(h)$ to $b$, then advance $h_{\varphi_k} \leftarrow h_{\varphi_k}/hh - 1/(i+1)!$ growing the factorial.

The scalar $B(h)$ multiplies the entire high-order correction and is genuinely free: any nonzero choice gives a consistent method and only changes the error constant and the conditioning. Two natural choices are $B(h) = hh$ ("bh1"), the simplest, and $B(h) = e^{hh} - 1 = h_{\varphi_1}$ ("bh2"), which matches the exponential weight of the integral more closely. At a tight budget where $h$ is large, bh2 tracks the true integrand better, so it is the default; I keep $B(h)$ as a named knob because it is exactly the kind of free constant a later robustness sweep would want to touch.

With $R$, $b$, and $B(h)$ in hand, predictor and corrector are the *same* formula evaluated with different amounts of information. The predictor, UniP, has the past points but not yet an evaluation at the new point $t$, so it solves the *reduced* system: for order $p$ it uses the $(p-1)$-dimensional solve $\rho_p = \text{solve}(R_{[:-1,:-1]}, b_{[:-1]})$ — the special case $p=2$ just gives $\rho_p = 0.5$ — and the update is
$$x_t = x_t^{(\text{base})} - \alpha_t\, B(h)\, \sum_k \rho_{p,k}\, D1_k.$$
The corrector, UniC, is applied *after* the predictor step has been taken and the network has been evaluated at the predicted point to get $m_t = x_\theta(x_t^{\text{pred}}, t)$. Now there is one *more* usable evaluation, so the corrector solves the *full* $p$-dimensional system $\rho_c = \text{solve}(R, b)$ — order 1 gives $\rho_c = 0.5$ — and refines the *same* base step with the extra difference $D1_t = m_t - m_0$:
$$x_t = x_t^{(\text{base})} - \alpha_t\, B(h)\,\Big(\sum_k \rho_{c,k}\, D1_k + \rho_{c,\text{last}}\, D1_t\Big),$$
where $x_t^{(\text{base})}$ and $m_0$ here are taken from the *previous* step's quantities — the corrector is refining the step that produced the point at which $m_t$ was just evaluated. So the loop is: at step $i$, first run the corrector on step $i-1$ using the model output just computed (free), then run the predictor for step $i$. The corrector raises the realized order by one over the predictor at the same NFE — this is precisely the "reuse the next step's evaluation" structure, made exact.

A few realities make it robust in practice. The $\varphi$ and $B(h)$ factors are small-argument exponentials, so `expm1` is used throughout to avoid catastrophic cancellation. The order ramps up as history accumulates — order 1 on the first step, then 2, then up to the configured maximum — and a `lower_order_final` option drops the order on the last step or two, where there is no future evaluation left to correct with (and where the trajectory is nearly straight at low noise anyway). The time grid is the EDM/Karras power schedule ($\rho = 7$) or uniform-$\lambda$; either concentrates the budget where the truncation error lives. For latent-space text-to-image there is no $[-1,1]$ bound, so thresholding is off and only the numerics matter. And the data prediction itself folds classifier-free guidance — the combined conditional and unconditional passes — inside the model wrapper, so the solver sees a clean $x_\theta$ oracle. The net result runs at one network call per step, the same NFE as DPM-Solver++(kM), but with each step's leading error corrected, which is exactly what wins in the 5–10 step regime.

```python
import torch


class UniPCBH:
    def __init__(self, ns, predict, solver_type="bh2", predict_x0=True):
        self.ns, self.predict = ns, predict
        self.solver_type, self.predict_x0 = solver_type, predict_x0

    def _R_b(self, rks, hh, B_h, order):
        h_phi_1 = torch.expm1(hh)
        R, b, h_phi_k, fact = [], [], h_phi_1 / hh - 1, 1
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * fact / B_h)
            fact *= i + 1
            h_phi_k = h_phi_k / hh - 1 / fact
        return torch.stack(R), torch.stack(b), h_phi_1

    def _bh(self, hh):
        return hh if self.solver_type == "bh1" else torch.expm1(hh)

    def _ratios(self, x, s0, m_list, lam_list, lam_s0, h, order):
        rks, D1s = [], []
        for i in range(1, order):
            rk = (lam_list[-(i + 1)] - lam_s0) / h
            rks.append(rk)
            D1s.append((m_list[-(i + 1)] - m_list[-1]) / rk)
        rks.append(torch.ones((), device=x.device))
        return torch.stack(rks), D1s

    def predictor(self, x, s0, t, m_list, lam_list, order):
        ns = self.ns
        lam_s0, lam_t = ns.lam(s0), ns.lam(t)
        h = lam_t - lam_s0
        hh = -h if self.predict_x0 else h
        B_h, m0 = self._bh(hh), m_list[-1]
        rks, D1s = self._ratios(x, s0, m_list, lam_list, lam_s0, h, order)
        R, b, h_phi_1 = self._R_b(rks, hh, B_h, order)
        x_base = (ns.sigma(t) / ns.sigma(s0)) * x - ns.alpha(t) * h_phi_1 * m0
        if D1s:
            D1s = torch.stack(D1s, dim=1)
            rhos_p = (torch.tensor([0.5], dtype=x.dtype, device=x.device) if order == 2
                      else torch.linalg.solve(R[:-1, :-1], b[:-1]))
            pred_res = torch.einsum("k,bk...->b...", rhos_p, D1s)
        else:
            pred_res = 0
        return x_base - ns.alpha(t) * B_h * pred_res

    def corrector(self, x_prev, s0, t, m_list, lam_list, model_t, order):
        ns = self.ns
        lam_s0, lam_t = ns.lam(s0), ns.lam(t)
        h = lam_t - lam_s0
        hh = -h if self.predict_x0 else h
        B_h, m0 = self._bh(hh), m_list[-1]
        rks, D1s = self._ratios(x_prev, s0, m_list, lam_list, lam_s0, h, order)
        R, b, h_phi_1 = self._R_b(rks, hh, B_h, order)
        x_base = (ns.sigma(t) / ns.sigma(s0)) * x_prev - ns.alpha(t) * h_phi_1 * m0
        rhos_c = (torch.tensor([0.5], dtype=x_prev.dtype, device=x_prev.device) if order == 1
                  else torch.linalg.solve(R, b))
        D1_t = model_t - m0
        if D1s:
            D1s = torch.stack(D1s, dim=1)
            corr_res = torch.einsum("k,bk...->b...", rhos_c[:-1], D1s)
        else:
            corr_res = 0
        return x_base - ns.alpha(t) * B_h * (corr_res + rhos_c[-1] * D1_t)

    @torch.no_grad()
    def sample(self, x, sigmas, max_order=3, lower_order_final=True):
        ns, N = self.ns, len(sigmas) - 1
        m_list, lam_list, x_prev, s_prev = [], [], None, None
        for i in range(N):
            sigma = sigmas[i]
            model_t = self.predict(x, sigma)                       # only call this step
            if x_prev is not None:
                order_c = min(max_order, len(m_list))      # past points available for the corrector
                x = self.corrector(x_prev, s_prev, sigma, m_list, lam_list, model_t, order_c)
            m_list.append(model_t)
            lam_list.append(ns.lam(sigma))
            if i == N - 1:
                x = m_list[-1]
                break
            order_p = min(max_order, len(m_list))
            if lower_order_final:
                order_p = min(order_p, N - i)
            x_prev, s_prev = x, sigma
            x = self.predictor(x, sigma, sigmas[i + 1], m_list, lam_list, order_p)
        return x
```
