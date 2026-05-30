# Rectified Adam (RAdam)

## Problem

Adaptive optimizers in the Adam family divide each coordinate's step by the square root of an
exponential-moving-average (EMA) estimate of the squared gradient. Early in training that estimate is
formed from very few gradient samples, so the adaptive learning rate `1/√v̂_t` is a high-variance
random quantity — at the first step its variance is literally divergent. The result is occasional
gigantic early updates that push the model into a bad region (the gradient distribution gets distorted
within ~10 steps), which is why adaptive methods empirically need a hand-tuned **learning-rate warmup**
to train stably. RAdam removes the need for warmup by rectifying that variance directly.

## Key idea

Model the gradients early in training as i.i.d. `N(0, σ²)`. Then `ψ²(.) = (1−β₂^t)/((1−β₂)Σβ₂^{t−i}g_i²)`
(the bias-corrected inverse second moment) is approximately scaled-inverse-chi-square with `ρ` degrees
of freedom, `ρ` being the effective number of samples. The variance of the adaptive rate `ψ` is finite
only for `ρ>4` and decreases monotonically as `ρ` grows. Estimate `ρ` at step `t` by matching the EMA
to a simple moving average with the same center of mass, then rectify the rate with a multiplier that
holds its variance constant across `t`. When the variance is not yet well-defined (`ρ_t ≤ 4`), drop the
adaptive denominator and take a plain momentum step.

## Final algorithm

Precompute `ρ_∞ = 2/(1−β₂) − 1` (maximum effective SMA length). At each step `t`:

```
g_t   = ∇f_t(θ_{t-1})
v_t   = β₂ v_{t-1} + (1−β₂) g_t²            # EMA second moment
m_t   = β₁ m_{t-1} + (1−β₁) g_t             # EMA first moment
m̂_t   = m_t / (1 − β₁^t)                    # bias-corrected first moment
ρ_t   = ρ_∞ − 2 t β₂^t / (1 − β₂^t)         # effective degrees of freedom

if ρ_t > 4:                                  # variance tractable
    l_t = sqrt( (1 − β₂^t) / v_t )           # bias-corrected adaptive rate
    r_t = sqrt( (ρ_t−4)(ρ_t−2)ρ_∞ / ((ρ_∞−4)(ρ_∞−2)ρ_t) )   # variance rectification
    θ_t = θ_{t-1} − α_t · r_t · m̂_t · l_t   # rectified adaptive step
else:                                        # variance not well-defined
    θ_t = θ_{t-1} − α_t · m̂_t               # un-adapted momentum (SGD-with-momentum) step
```

Key facts:
- `ρ_t` rises monotonically from 1 (at `t=1`, one effective sample) toward `ρ_∞`; the correction
  `2tβ₂^t/(1−β₂^t) → 0` as `t→∞`.
- `r_t = sqrt(Var[ψ]|_{ρ_∞} / Var[ψ]|_{ρ_t}) ≤ 1`, rising to 1 as training proceeds — the same
  rising shape as a linear warmup `min(t,T_w)/T_w`, but with no `T_w` to tune. It is derived using the
  stable first-order variance approximation `Var[ψ] ≈ ρ/(2(ρ−2)(ρ−4)σ²)` (the exact Beta-function form
  is numerically unstable).
- If `β₂ ≤ 0.6` then `ρ_∞ ≤ 4`, so the rectified branch never activates and RAdam reduces exactly to
  SGD with momentum: the second-moment estimate never gains enough effective samples to be trusted.

## Code

```python
import math
import torch
from torch.optim.optimizer import Optimizer

class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(betas))
        self.degenerated_to_sgd = degenerated_to_sgd
        # cache (t, rho_t, step_size) keyed by t % 10; rho_t depends only on (t, beta2)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                p_fp32 = p.data.float()

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_fp32)      # m_t
                    state['exp_avg_sq'] = torch.zeros_like(p_fp32)   # v_t
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_fp32)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # EMA second / first moments
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                t = state['step']

                buffered = group['buffer'][int(t % 10)]
                if t == buffered[0]:
                    rho_t, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = t
                    beta2_t = beta2 ** t
                    rho_inf = 2 / (1 - beta2) - 1                       # rho_infinity
                    rho_t = rho_inf - 2 * t * beta2_t / (1 - beta2_t)   # effective DoF at step t
                    buffered[1] = rho_t
                    # >=5 (rather than >4): conservative margin since rho_t is an approximation
                    if rho_t >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (rho_t - 4) / (rho_inf - 4)
                            * (rho_t - 2) / rho_t
                            * rho_inf / (rho_inf - 2)
                        ) / (1 - beta1 ** t)
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** t)              # un-adapted momentum
                    else:
                        step_size = -1                                  # skip
                    buffered[2] = step_size

                if rho_t >= 5:
                    if group['weight_decay'] != 0:
                        p_fp32.add_(p_fp32, alpha=-group['weight_decay'] * group['lr'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    p.data.copy_(p_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_fp32.add_(p_fp32, alpha=-group['weight_decay'] * group['lr'])
                    p_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                    p.data.copy_(p_fp32)

        return loss
```

Usage: drop-in replacement for Adam, no warmup schedule required.
```python
optimizer = RAdam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
```
