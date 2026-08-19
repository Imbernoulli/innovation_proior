**Problem.** Both single-constant readings (rung 2) beat the restricted-tree search (rung 1) but
neither clears AdamW — a single β is forced to simultaneously serve two jobs, remembering gradient
history (wants β near 1) and staying reactive to the current gradient before signing (wants real
weight on the fresh gradient).

**Key idea.** Go back to the raw discovered program's *actual* two chained `interp` calls (with
different constants, ≈0.9 and ≈1.1) instead of the single-buffer simplification tested last rung.
Substituting one into the other algebraically (`m_t = interp(g_t, v_{t-1}, b1)` into
`v_t = interp(g_t, m_t, b2)`) collapses to `v_t = 0.01·g_t + 0.99·v_{t-1}` — an exact single EMA at
β≈0.99 on the raw gradient stream (coefficients sum to exactly 1 by construction of composed
affine interpolations; holds at full unrounded precision too, β≈0.99822). So only *one* buffer
needs to persist — at the slow rate — while the quantity that actually gets signed is a *different*,
never-stored, fast blend of that buffer with the fresh gradient. Memory (β₂) and reactivity (β₁)
each get their own dial instead of fighting over one. Also drop: `cosh` (assigns into a variable
overwritten before ever being read — dead by data-flow, not by ablation), and `clip`/`arcsin` on
the incoming gradient (ablated, no quality drop).

**Rung-3 fill (final).**

```
cₜ = β₁·mₜ₋₁ + (1−β₁)·gₜ                  # interpolated momentum used for the step
θₜ = θₜ₋₁ − ηₜ·( sign(cₜ) + λ·θₜ₋₁ )       # signed step + decoupled weight decay
mₜ = β₂·mₜ₋₁ + (1−β₂)·gₜ                  # momentum buffer, its own slower EMA
```

β₁ = 0.9, β₂ = 0.99 (the search's own constants). Sign output is ±1 per coordinate, larger norm
than AdamW's or either ablation's step at the same nominal lr, so lr is tuned down and, since
effective decoupled weight decay is lr·λ, λ is let move opposite lr in the sweep.

```python
from __future__ import annotations
from typing import Tuple, Callable
import torch
from torch.optim.optimizer import Optimizer


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # decoupled weight decay: theta <- theta * (1 - lr*lambda)
    p.data.mul_(1. - lr * wd)

    # the step: sign of the interpolated momentum  c = beta1*m + (1-beta1)*g
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1. - beta1).sign_()
    p.add_(update, alpha=-lr)                      # theta <- theta - lr * sign(c)

    # the momentum buffer updates on its OWN, slower constant beta2 (decoupled from the step)
    exp_avg.mul_(beta2).add_(grad, alpha=1. - beta2)


class Lion(Optimizer):
    def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.update_fn = update_fn

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group['params']):
                grad, lr, wd, beta1, beta2, state = (
                    p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]
                )
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                self.update_fn(p, grad, state['exp_avg'], lr, wd, beta1, beta2)
        return loss
```

**Why this rung.** It is the one structure on the ladder where memory and reactivity each get their
own dial, which the algebra shows the raw discovered program actually specified — the previous
rung's single-buffer read had thrown that structure away as apparent noise.

**What to watch.** Both Ablation variants (78.23/84.28/66.13, 78.19/84.17/65.96 on ViT-S/16;
79.54/85.10/68.07, 79.90/85.36/68.20 on ViT-B/16) and AdamW (78.89/84.61/66.73; 80.12/85.46/68.14)
— a real, consistent margin over both ablations at both scales would confirm the mechanism;
whether that margin also clears AdamW decides whether this is the final rung.
