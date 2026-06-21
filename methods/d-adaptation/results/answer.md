# D-Adaptation: learning-rate-free convex optimization

## Problem

Minimize a convex, G-Lipschitz f over R^p with the subgradient / dual-averaging method. The
worst-case-optimal step size γ = D/(G√n) needs the distance to a solution, D = ‖x₀ − x*‖, which is
unknown because it depends on x*. Adaptivity to G is solved by AdaGrad-Norm step sizes; the remaining
obstacle is the unknown D in the numerator. D-Adaptation removes the learning-rate hyperparameter
entirely — no line search, no extra function-value or gradient evaluations per step, no log-factor
penalty in the asymptotic rate.

## Key idea

A convergence proof gives an *upper* bound on suboptimality that is linear in D (use Cauchy–Schwarz
on ⟨s_{n+1}, x₀−x*⟩ instead of completing the square). Since suboptimality is nonnegative, that bound
**inverts into a lower bound on D** built from observed quantities. The raw lower bound can go negative
once progress is fast, so maintain the running maximum of all bounds seen: it is monotone, provably
stays ≤ D, and bootstraps from a tiny seed d₀ up toward D. Use this estimate as the dual-averaging
weight; the d-weighted average iterate makes the final rate independent of how high d climbed.

## Core algorithm (Dual Averaging with D-Adaptation)

Input: x₀, d₀ > 0 (default 1e-6). s₀ = 0, g₀ ∈ ∂f(x₀), γ₀ = 1/‖g₀‖. If g₀ = 0, return x₀.

For k = 0, 1, …, n:
- g_k ∈ ∂f(x_k)
- s_{k+1} = s_k + d_k g_k
- γ_{k+1} = 1 / √( Σ_{i=0}^{k} ‖g_i‖² )
- Lower-bound estimate (two equivalent options for SGD):
  - Option I:  d̂_{k+1} = ( γ_{k+1} ‖s_{k+1}‖² − Σ_{i=0}^{k} γ_i d_i² ‖g_i‖² ) / ( 2 ‖s_{k+1}‖ )
  - Option II: d̂_{k+1} = ( Σ_{i=0}^{k} d_i γ_i ⟨g_i, s_i⟩ ) / ‖s_{k+1}‖   (numerator ≥ Option I)
- d_{k+1} = max( d_k, d̂_{k+1} )          (the lower-bound ratchet)
- x_{k+1} = x₀ − γ_{k+1} s_{k+1}

Return x̂_n = ( Σ_{k=0}^n d_k x_k ) / ( Σ_{k=0}^n d_k ).

## Guarantees

- **Lower bound (Theorem).** For all n, D ≥ d̂_{n+1}, hence d_k ≤ D for all k (given d₀ ≤ D).
- **Asymptotic rate.** f(x̂_n) − f* = O(DG/√(n+1)) — the worst-case-optimal rate, with no log factor,
  with no knowledge of D. Explicitly ≤ 12DG/√(n+1) + 8DG²/((n+1)‖g₀‖).
- **Non-asymptotic rate.** With γ_{k+1} = 1/√(G² + Σ‖g_i‖²) and returning x̂_t at
  t = argmin_{k≤n} d_{k+1}/(Σ_{i≤k} d_i), for n ≥ 2log₂(D/d₀),
  f(x̂_t) − f* ≤ 16 DG log_{2+}(D/d₀)/√(n+1), where log_{2+}(x)=max(1, log₂ x).
  The price of not knowing D is a factor log(1+D/d₀) (vs D/d₀ for a naive d₀-proportional step), so d₀
  is not a hyperparameter.
- **Where d settles (Theorem).** If x_n → x*, then lim_n d_n ≥ D/(1+√3) ≈ 0.366 D (optimal Young split
  θ = 1+√3). d need not reach D; any constant fraction gives the optimal rate.
- **Gradient-descent variant** incurs an extra log(n+2) factor (the generic any-time-step penalty on
  unbounded domains), which dual averaging avoids; practical performance is nearly identical.
- **Coordinate-wise (ℓ∞) D-Adapted AdaGrad** attains O(p G∞ D∞/√(n+1)) without knowing D∞.

## Practical optimizers

Drop-in PyTorch optimizers. Set `lr = 1.0` (it is a multiplier on the D-adapted rate, used to carry
the problem's usual learning-rate *schedule* with base value 1.0; warmup increases toward 1.0,
annealing decreases from 1.0). `d0` rarely needs changing. `growth_rate` optionally caps how fast d
may grow per step (default ∞; values like 1.02 give a warmup-like effect / stabilize). The SGD variant
uses the hyper-gradient estimate (Option II) with a practical factor of 2 (the rate is invariant to a
constant step multiplier); Adam needs no such constant.

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

Usage: replace your optimizer with `DAdaptSGD(model.parameters())` or
`DAdaptAdam(model.parameters(), decouple=True)`, keep `lr=1.0`, and keep the same learning-rate
*scheduler* you would normally use on the problem. The base learning rate is then set automatically by
the distance estimate d; inspect `d * lr` to see the effective rate in use.
