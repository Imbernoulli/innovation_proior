# NAdam (Nesterov-accelerated adaptive moment estimation)

## Problem

Adam combines two ideas: a momentum term (a decaying mean of the gradient) and an adaptive
per-coordinate learning rate (a decaying mean of the squared gradient). The adaptive part is what makes
it robust; the momentum part is the *classical* kind, which commits to the old velocity direction
before consulting the current gradient. Nesterov's accelerated gradient corrects that step with a
look-ahead gradient — a better direction and a provably better convergence rate on smooth convex
problems. NAdam carries the accelerated momentum *inside* Adam, leaving the adaptive rescaling
untouched.

## Key idea

Replace Adam's classical first-moment step with the simplified-Nesterov step (Sutskever et al. 2013):
advance the momentum step by one so it uses the freshly updated first moment (which already contains
the current gradient), folding the look-ahead into the *update* rather than the *gradient evaluation
point*. The gradient is still differentiated at the current parameters θₜ₋₁ — no shadow look-ahead
parameters — but the step now reacts to the current gradient twice (once directly, once through the
advanced moment). The second-moment adaptive denominator √n̂ₜ + ε is unchanged.

## Algorithm

State per parameter: first moment m, second moment n (both init 0), step counter t, and the running
momentum product. The stored first moment remains the ordinary β₁ EMA; μ_t and μ_{t+1} are scheduled
weights for the Nesterov blend and its normalizers. With learning rate α, betas (β₁, β₂), ε, weight
decay λ, and momentum-schedule rate ψ:

```
t ← t + 1
μ_t   = β₁·(1 − ½·0.96^{t·ψ})           # momentum warms from ~β₁/2 toward β₁
μ_{t+1} = β₁·(1 − ½·0.96^{(t+1)·ψ})
mu_product ← mu_product · μ_t            # running product for bias correction

# with decoupled_weight_decay=True, shrink weights before the moment updates:
θ ← (1 − αλ)·θ

m ← β₁·m + (1−β₁)·g
n ← β₂·n + (1−β₂)·g²
denom = √(n / (1 − β₂^t)) + ε

# simplified-Nesterov step = fresh-gradient piece + advanced-momentum piece:
θ ← θ − α·(1−μ_t)·g  / (1 − mu_product)          / denom
θ ← θ − α·μ_{t+1}·m / (1 − mu_product·μ_{t+1})   / denom
```

The two bias-correction denominators differ because the momentum piece is *advanced* by one step
(divided by 1 − Π_{i=1}^{t+1} μ_i) while the fresh-gradient piece keeps 1 − Π_{i=1}^{t} μ_i. If the
scheduled blend is replaced by Adam's single m/(1−β₁^t) first-moment term, the update reduces to
ordinary Adam; with decoupled weight decay, ordinary AdamW.

## Hyperparameters

- α: learning rate (default 2e-3).
- betas = (β₁, β₂): first/second-moment decays (default β₁ = 0.9, β₂ = 0.999).
- ε = 1e-8 numerical floor in the denominator.
- weight_decay λ with `decoupled_weight_decay=True` for AdamW-style direct shrinkage.
- momentum_decay ψ = 4e-3 (the schedule rate; default).

## Working code

The update lines mirror the non-capturable single-tensor `torch.optim.NAdam` branch: the running
mu-product schedule, the two additive bias-corrected `addcdiv_` pieces, and the
`decoupled_weight_decay` shrink before the moment updates.

```python
import torch
from torch.optim.optimizer import Optimizer


class NAdam(Optimizer):
    """Adam with Nesterov-accelerated momentum (the look-ahead absorbed into the
    update), with optional decoupled (AdamW-style) weight decay."""

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, momentum_decay=4e-3, decoupled_weight_decay=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        momentum_decay=momentum_decay,
                        decoupled_weight_decay=decoupled_weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            psi = group["momentum_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["mu_product"] = torch.tensor(1.0)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"].add_(1)
                t = int(state["step"].item())
                m, n = state["exp_avg"], state["exp_avg_sq"]

                mu_t  = beta1 * (1.0 - 0.5 * (0.96 ** (t * psi)))
                mu_t1 = beta1 * (1.0 - 0.5 * (0.96 ** ((t + 1) * psi)))
                state["mu_product"].mul_(mu_t)
                mu_product = float(state["mu_product"].item())

                if wd != 0:
                    if group["decoupled_weight_decay"]:
                        p.mul_(1 - lr * wd)
                    else:
                        g = g.add(p, alpha=wd)

                m.lerp_(g, 1 - beta1)
                n.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                denom = (n / (1 - beta2 ** t)).sqrt_().add_(eps)

                p.addcdiv_(g, denom, value=-lr * (1 - mu_t) / (1 - mu_product))
                p.addcdiv_(m, denom, value=-lr * mu_t1 / (1 - mu_product * mu_t1))
        return loss
```
