The thing that actually wastes my time in convex optimization is a single number. For a convex, $G$-Lipschitz function $f$ on $\mathbb{R}^p$ and the subgradient method $x_{k+1} = x_k - \gamma_k g_k$, the entire behaviour rides on the step size $\gamma_k$, and the theory tells me, annoyingly precisely, what the best constant step is: $\gamma = D/(G\sqrt{n})$, where $D = \|x_0 - x^*\|$ is the distance from the start to a solution. Plug it in and I get the worst-case-optimal rate $f(\hat x_n) - f^* = O(DG/\sqrt{n})$ for this class. So in principle there is nothing to tune — the optimal step is a formula. The catch is that the formula contains $D = \|x_0 - x^*\|$, which depends on $x^*$, the very thing I am trying to compute. It is circular: I cannot know the distance to the answer before I have the answer. The $G$ in the denominator I am not worried about, because the AdaGrad-Norm step $\gamma_k = D/\sqrt{\sum_{i\le k}\|g_i\|^2}$ self-calibrates to the gradient scale (the load-bearing fact being $\sum_k \gamma_k \|g_k\|^2 \le 2\sqrt{\sum_k\|g_k\|^2}$ with $\gamma_k = 1/\sqrt{\sum_{i<k}\|g_i\|^2}$). What is left, every single time, is the numerator $D$ — and in practice nobody knows it, so people grid-search a log-spaced range of step scales and train the model a dozen times. That grid search is the real cost of "tuning the learning rate."

The existing options all dodge the unknown $D$ rather than computing it. Polyak's step $\gamma_k = (f(x_k) - f^*)/\|g_k\|^2$ is optimal with no log factor, but it trades "don't know $D$" for "don't know $f^*$" — same disease, different organ, and estimating $f^*$ online is unstable. Exact line search gives the optimal rate with no constants but costs a line search per step. Coin-betting / COCOB achieves the best regret possible without $D$, but it is a $\sqrt{\log}$ worse than knowing $D$ and, fatally, it bakes in its own implicit schedule so it cannot accept the warmup-then-cosine schedules transformers need. DoG estimates the distance by how far it has moved, $\bar r_k = \max_{i\le k}\|x_i - x_0\|$, but that proxy is not guaranteed bounded — there is a convex example where it diverges — so it needs dampening and pays extra log factors. Carmon–Hinder come closest: they characterize the ideal step as a fixed point $\eta = \varphi(\eta)$, $\varphi(\eta) = \|x_0 - x^*\|/\sqrt{\sum\|g_i(\eta)\|^2}$, and bisect, getting optimal-up-to-loglog — but it is still a search wrapping the optimizer with a residual loglog factor. None of them just computes a usable value of $D$ from the run itself.

I propose D-Adaptation. The blunt question is whether any quantity I observe during the run is provably $\le D$, because a certified lower bound on $D$ is exactly what I need: underestimating $D$ makes my step too small, so I am slow but I never overshoot — a far friendlier failure mode than DoG's possibly-unbounded estimate. Where would such a bound come from? Convergence proofs are full of *upper* bounds on suboptimality written in terms of $D$, and an upper bound on a nonnegative quantity, rearranged, is a *lower* bound on whatever sits inside it. So I run the convergence proof backwards: not to certify convergence, but to certify a value of $D$. I work in dual averaging, keeping a weighted gradient sum $s_{k+1} = s_k + \lambda_k g_k$ with positive weights $\lambda_k$, and $x_{k+1} = x_0 - \gamma_{k+1} s_{k+1}$. Starting from convexity and splitting $x_k - x^* = (x_k - x_0) + (x_0 - x^*)$, the key move is to apply Cauchy–Schwarz to $\langle s_{n+1}, x_0 - x^*\rangle$ rather than completing the square — this keeps $D$ *linear* (as $D\|s_{n+1}\|$) instead of producing a $D^2$ term. Running the standard dual-averaging telescoping on the inner-product sum $-\sum_k \gamma_k \lambda_k \langle g_k, s_k\rangle$, and using that $\gamma_k$ is nonincreasing so the $\frac12\sum(\gamma_{k+1}-\gamma_k)\|s_{k+1}\|^2$ term is $\le 0$ and can be dropped, gives

$$\sum_k \lambda_k (f(x_k) - f^*) \le D\,\|s_{n+1}\| + \sum_k \tfrac12 \gamma_k \lambda_k^2 \|g_k\|^2 - \tfrac12 \gamma_{n+1}\|s_{n+1}\|^2.$$

This is exactly what I wanted — linear in $D$, and it even picks up a free negative $-\frac12\gamma_{n+1}\|s_{n+1}\|^2$ term out of the telescoping. (The trade is legitimate: $\frac12\gamma^{-1}D^2 - (D\|s\| - \frac12\gamma\|s\|^2) = \frac12\gamma^{-1}(D - \gamma\|s\|)^2 \ge 0$, with equality when $D = \|x_0 - x_{n+1}\|$, so this is a genuine tightening of the classical $\frac12\gamma^{-1}D^2 + \sum\frac12\gamma\lambda^2\|g\|^2$ bound.) Now invert: the left side is a sum of nonnegative terms ($f(x_k)\ge f^*$, $\lambda_k>0$), so it is $\ge 0$, hence the right side is $\ge 0$, and solving for $D$ gives

$$D \ge \hat d_{n+1} := \frac{\gamma_{n+1}\|s_{n+1}\|^2 - \sum_{k}\gamma_k \lambda_k^2 \|g_k\|^2}{2\,\|s_{n+1}\|}.$$

Everything on the right is observed during the run — the weighted gradient sum, its norm, the gradient norms, the step sizes. No $x^*$, no $f^*$, no true $D$. I have manufactured a certified lower bound on the distance to the solution out of the convergence proof.

The wall is that $\hat d$ is not directly usable: the numerator $\gamma_{n+1}\|s_{n+1}\|^2 - \sum_k \gamma_k \lambda_k^2\|g_k\|^2$ has no reason to be positive. Early on, when $s$ is small and the accumulated $\sum\gamma\lambda^2\|g\|^2$ dominates, $\hat d$ goes negative — a meaningless distance, the bound gone vacuous. This happens precisely when progress is already fast (the gradients are starting to cancel), so it is the certificate honestly saying "you don't need a bigger $D$ estimate right now." The fix is the natural one: a lower bound stays true forever, so I keep the best certificate I have ever seen,

$$d_{k+1} = \max(d_k,\ \hat d_{k+1}),\qquad d_0 > 0 \text{ small}.$$

This running maximum is nondecreasing, stays $\le D$ (every $\hat d \le D$, and the seed $d_0 \le D$ is harmless), and ignores the vacuous negative certificates. The negative-$\hat d$ problem evaporates. And crucially it bootstraps: each time the step grows, $s$ and $\|s\|^2$ grow, which makes the next $\hat d$ bigger, which makes the next step bigger — so from an arbitrarily tiny seed $d_0$ it climbs exponentially toward $D$ and parks below it. I lock down the weights with $\lambda_k = d_k$: this makes the left-side suboptimality a $d$-weighted average $\sum_k d_k (f(x_k)-f^*)$ that counts the later, better-estimated steps more, and it makes the step magnitude scale as $d_k/\sqrt{\sum\|g\|^2}$ — exactly the AdaGrad-Norm step with my estimated $D$ in the numerator, which is the whole point. The seed $d_0$ is not a hyperparameter: I set it tiny (default $10^{-6}$, even $10^{-16}$ works, the only floor being float16 underflow), and it gets bootstrapped away.

What makes this actually achieve the optimal rate is the $d$-weighted average iterate $\hat x_n = (\sum d_k x_k)/(\sum d_k)$. Bounding $\|s_{n+1}\|$ via Young's inequality ($2d_{n+1}\|s_{n+1}\| \le 2d_{n+1}^2/\gamma_{n+1} + \frac12\gamma_{n+1}\|s_{n+1}\|^2$, then substituting the inverted bound) gives $\|s_{n+1}\| \le 2d_{n+1}/\gamma_{n+1} + (\sum\gamma_k\lambda_k^2\|g\|^2)/(2d_{n+1})$, and chaining back into the suboptimality bound with $\lambda_k = d_k \le d_{n+1} \le D$ yields the key intermediate $\sum_k d_k(f-f^*) \le 2D d_{n+1}\sqrt{\sum\|g\|^2} + D d_{n+1}\sum_k\gamma_k\|g_k\|^2$. Dividing by $\sum_k d_k$ is where the $d$-weighting earns its keep: once $d$ has climbed to within a constant factor of its limit over a constant fraction of the run (it converges, being nondecreasing and bounded by $D$), $\sum_k d_k \ge \frac14(n+1)d_{n+1}$, the $d_{n+1}$ cancels, and using $\sum\gamma_k\|g_k\|^2 \le 2\sqrt{\sum\|g\|^2}$ and $\sqrt{\sum\|g\|^2}\le G\sqrt{n+1}$,

$$f(\hat x_n) - f^* \le \frac{16\,DG}{\sqrt{n+1}} + O\!\left(\frac{DG^2}{(n+1)\|g_0\|}\right) = O\!\left(\frac{DG}{\sqrt{n+1}}\right).$$

The optimal rate, asymptotically, with no log factor, no knowledge of $D$, no line search, no extra evaluations. The asymptotic caveat is honest — for a fixed horizon an adversary can make $d$ crawl and only reach $D$ at the last step — but the price is small: with $\gamma_{k+1} = 1/\sqrt{G^2 + \sum_{i\le k}\|g_i\|^2}$ and returning $\hat x_t$ at $t = \arg\min_{k\le n} d_{k+1}/(\sum_{i\le k} d_i)$, a lemma that a nondecreasing sequence bounded by $D$ can only double about $\log_2(D/d_0)$ times gives $f(\hat x_t) - f^* \le 16\,DG\,\log_{2+}(D/d_0)/\sqrt{n+1}$ for $n \ge 2\log_2(D/d_0)$, where $\log_{2+}(x) = \max(1, \log_2 x)$. So the cost of not knowing $D$ is a factor $\log(1 + D/d_0)$ — logarithmic, versus the $D/d_0$ a naive $d_0$-proportional step would pay — confirming $d_0$ is essentially free. I choose dual averaging over plain gradient descent deliberately: the GD version incurs an extra $\log(n+2)$ factor (the generic any-time-step penalty on an unbounded domain, from $\sum \|g_k\|^2/(G^2+\sum\|g_i\|^2) \le \log(n+2)$), which DA dodges at no practical cost. Under a mild convergence assumption $d$ settles at no less than about a third of $D$: redoing the Young split with a tunable $\theta$ and optimizing at $\theta^* = 1+\sqrt3$ gives $\gamma_n\|s_n\| \le (1+\sqrt3)d_n$, hence $\lim_n d_n \ge D/(1+\sqrt3) \approx 0.366\,D$ — and any constant fraction of $D$ suffices because the $d$-weighted average cancels $d_{n+1}$ anyway.

There is one more piece worth flagging for the deep-learning versions. The same telescoping shows $\sum_k \gamma_k\lambda_k\langle g_k, s_k\rangle \ge \frac12\gamma_{n+1}\|s_{n+1}\|^2 - \sum_k\frac12\gamma_k\lambda_k^2\|g_k\|^2$, which means the alternative estimate $\hat d_{n+1} = (2\sum_k\gamma_k\lambda_k\langle g_k,s_k\rangle)/\|s_{n+1}\|$ (Option II) has a numerator provably $\ge$ Option I's, with equality in the flat-weight SGD case. The quantity $\langle g_k, s_k\rangle$ is the inner product of the gradient with the current step direction — the (negative) hyper-gradient — but where prior hyper-gradient methods used only its *sign* (agree → bump up, disagree → down, with an extra hyper-learning-rate to tune), here it estimates the *magnitude* of the right step, and because it feeds a lower bound protected by the running max, I can freely impose a decreasing schedule on top without the signal fighting back. That schedule-compatibility is what coin-betting cannot give. To make the optimizers real for deep learning, where the theory technically stops applying but the mechanism still bootstraps, I write SGD using Option II (with a deliberate factor of 2 — the rate is invariant to a constant step scale, and the larger estimate helps empirically) and Adam by re-expressing $s$, the gradient accumulators, and the inverted bound as exponential moving averages: with weights $\lambda_k = \sqrt{\beta_2^{-k}}$ the EMA companions $\hat v = \beta_2\hat v + (1-\beta_2)g^2$ and $\hat s = \sqrt{\beta_2}\,\hat s + (1-\sqrt{\beta_2})g$ make the un-weighted step $x - g/\sqrt{\hat v}$ exactly Adam's, and the bound collapses to $\hat d = (\|s\|^2_{A^{-1}}/(1-\beta_2) - \sum d^2\|g\|^2_{A^{-1}})/\|s\|_1$, the $(1-\beta_2)$ being the scale correction. Both are drop-in PyTorch optimizers that need no learning rate, accept any schedule (carried as a multiplier `lr` with base $1.0$), and add no per-step gradient or function evaluations.

```python
import torch
import math


class DAdaptSGD(torch.optim.Optimizer):
    """SGD with D-Adaptation automatic step sizes. Leave lr=1.0 unless unstable."""
    def __init__(self, params, lr=1.0, momentum=0.0, weight_decay=0.0,
                 d0=1e-6, growth_rate=float('inf')):
        if not 0.0 < d0: raise ValueError(f"Invalid d0: {d0}")
        if not 0.0 < lr: raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, k=0,
                        numerator_weighted=0.0, d=d0, growth_rate=growth_rate)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        group = self.param_groups[0]
        lr = max(g['lr'] for g in self.param_groups)
        decay, momentum, k = group['weight_decay'], group['momentum'], group['k']
        ck = 1 - momentum
        numerator_weighted = group['numerator_weighted']
        growth_rate, d = group['growth_rate'], group['d']

        # step 0: G ~= ||g0|| sets the denominator scale
        if k == 0:
            g_sq = 0.0
            for grp in self.param_groups:
                for p in grp['params']:
                    if p.grad is None:
                        continue
                    g = p.grad.data
                    if decay != 0:
                        g.add_(p.data, alpha=decay)
                    g_sq += (g * g).sum().item()
            group['g0_norm'] = math.sqrt(g_sq)
        g0_norm = group['g0_norm']

        dlr = d * lr / g0_norm

        sk_sq = 0.0
        delta_numerator = 0.0
        for grp in self.param_groups:
            for p in grp['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                st = self.state[p]
                if 'z' not in st:
                    st['z'] = torch.clone(p.data).detach()
                    st['s'] = torch.zeros_like(p.data).detach()
                    st['x0'] = torch.clone(p.data).detach()
                if decay != 0:
                    g.add_(p.data, alpha=decay)
                s = st['s']
                # hyper-gradient numerator <g_k, s_k>, before updating s (Option II)
                delta_numerator += dlr * torch.dot(g.flatten(), s.flatten()).item()
                s.add_(g, alpha=dlr)                       # s_{k+1} = s_k + dlr * g_k
                sk_sq += (s * s).sum().item()

        numerator_weighted += delta_numerator
        if sk_sq == 0:
            return loss
        if lr > 0.0:
            d_hat = 2 * numerator_weighted / math.sqrt(sk_sq)   # Option II x factor 2
            d = max(d, min(d_hat, d * growth_rate))             # lower-bound ratchet

        for grp in self.param_groups:
            grp['numerator_weighted'], grp['d'], grp['g0_norm'] = numerator_weighted, d, g0_norm
            for p in grp['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                st['z'].copy_(st['x0'] - st['s'])          # z = x0 - s
                p.data.mul_(1 - ck).add_(st['z'], alpha=ck)  # primal-averaging momentum
            grp['k'] = k + 1
        return loss


class DAdaptAdam(torch.optim.Optimizer):
    """Adam with D-Adaptation automatic step sizes. Leave lr=1.0 unless unstable.
    Set decouple=True for AdamW-style weight decay."""
    def __init__(self, params, lr=1.0, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decouple=False, d0=1e-6, growth_rate=float('inf')):
        if not 0.0 < d0: raise ValueError(f"Invalid d0: {d0}")
        if not 0.0 < lr: raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 < eps: raise ValueError(f"Invalid eps: {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        d=d0, k=0, gsq_weighted=0.0, decouple=decouple, growth_rate=growth_rate)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        group = self.param_groups[0]
        beta1, beta2 = group['betas']
        d = group['d']
        lr = max(g['lr'] for g in self.param_groups)
        dlr = d * lr
        growth_rate, decouple = group['growth_rate'], group['decouple']
        gsq_weighted = group['gsq_weighted']

        g_sq = 0.0
        sksq_weighted = 0.0
        sk_l1 = 0.0
        for grp in self.param_groups:
            decay, eps = grp['weight_decay'], grp['eps']
            for p in grp['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                if decay != 0 and not decouple:
                    g.add_(p.data, alpha=decay)
                st = self.state[p]
                if 'step' not in st:
                    st['step'] = 0
                    st['s'] = torch.zeros_like(p.data).detach()
                    st['exp_avg'] = torch.zeros_like(p.data).detach()       # m
                    st['exp_avg_sq'] = torch.zeros_like(p.data).detach()    # v-hat
                m, v = st['exp_avg'], st['exp_avg_sq']
                gg = g * g
                m.mul_(beta1).add_(g, alpha=dlr * (1 - beta1))            # weighted EMA of g
                v.mul_(beta2).add_(gg, alpha=1 - beta2)                   # Adam v-hat
                denom = v.sqrt().add_(eps)
                g_sq += gg.div_(denom).sum().item()                      # ||g||^2_{A^-1}
                s = st['s']
                s.mul_(beta2).add_(g, alpha=dlr * (1 - beta2))           # s EMA
                sksq_weighted += (s * s).div(denom).sum().item()         # ||s||^2_{A^-1}
                sk_l1 += s.abs().sum().item()                            # ||s||_1

        gsq_weighted = beta2 * gsq_weighted + g_sq * (dlr ** 2) * (1 - beta2)
        if sk_l1 == 0:
            return loss
        if lr > 0.0:
            # inverted-bound estimate (Option I, EMA / weighted-norm form)
            d_hat = (sksq_weighted / (1 - beta2) - gsq_weighted) / sk_l1
            d = max(d, min(d_hat, d * growth_rate))                      # lower-bound ratchet

        for grp in self.param_groups:
            grp['gsq_weighted'], grp['d'] = gsq_weighted, d
            decay, eps = grp['weight_decay'], grp['eps']
            for p in grp['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                st['step'] += 1
                m, v = st['exp_avg'], st['exp_avg_sq']
                denom = v.sqrt().add_(eps)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)              # AdamW-style decay
                p.data.addcdiv_(m, denom, value=-1)                     # x -= m / (sqrt(v)+eps)
            grp['k'] = group['k'] + 1
        return loss
```
