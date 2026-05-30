# Lion (EvoLved Sign Momentum)

**Problem.** State-of-the-art models are trained almost exclusively with hand-designed optimizers (AdamW, Adafactor), which carry two per-parameter buffers and were never themselves the product of a search. Lion is the result of formulating optimizer design as **program search** — representing update rules as short programs and discovering one by regularized evolution — and then simplifying the discovered rule into something analyzable.

**Key idea.** Track a *single* momentum buffer. Take the step as the **sign** of a linear interpolation between the current gradient and the momentum, and update the momentum with a *different, slower* interpolation constant. The two constants decouple how long the optimizer remembers gradients (β₂, the buffer) from how much weight the current gradient gets in the actual step (β₁, the interpolation that is signed). The sign gives every coordinate a uniform-magnitude update, which injects noise and acts as a regularizer, and uses one buffer instead of Adam's two.

**Algorithm.** Given β₁, β₂, weight decay λ, learning-rate schedule ηₜ, momentum m (init 0):

```
cₜ      = β₁·mₜ₋₁ + (1−β₁)·gₜ                  # interpolated momentum used for the step
θₜ      = θₜ₋₁ − ηₜ·( sign(cₜ) + λ·θₜ₋₁ )       # signed step + decoupled weight decay
mₜ      = β₂·mₜ₋₁ + (1−β₂)·gₜ                  # momentum buffer, its own slower EMA
```

As a program (the discovered, simplified form):

```python
def train(weight, gradient, momentum, lr):
    update   = interp(gradient, momentum, beta1)   # (1-beta1)*g + beta1*m
    update   = sign(update)
    momentum = interp(gradient, momentum, beta2)   # (1-beta2)*g + beta2*m
    update   = update + weight * weight_decay      # decoupled weight decay
    update   = update * lr
    return update, momentum                        # outer loop: weight = weight - update
```

**Defaults and tuning.** β₁ = 0.9, β₂ = 0.99 (discovered by the search; β₁ = 0.95, β₂ = 0.98 trades memory for stability). Both interpolation constants are necessary: collapsing to a single EMA `sign(interp(g, m, β))` with β = 0.9 or 0.99 underperforms. Because the sign update is element-wise ±1, its norm is larger than SGD/adaptive steps, so the learning rate should be **3–10× smaller** than AdamW's (scale initial/peak/final by the same ratio; leave schedule and clipping unchanged). Since the effective decoupled weight decay is lr·λ, the weight decay λ should be **3–10× larger** than AdamW's to hold lr·λ constant. Example: AdamW lr = 1e-3, λ = 1.0 → Lion lr = 1e-4, λ = 10.0.

**Why it works.**
- *Sign → uniform magnitude + regularization.* Discarding gradient magnitude makes every coordinate move by the same amount and injects noise into the update, which biases training toward flatter minima; the benefit tends to grow with batch size (where the averaged gradient's sign is reliable).
- *Decoupled β₁/β₂.* β₂ ≈ 0.99 gives the buffer a ~10× longer gradient memory than the usual 0.9, while β₁ ≈ 0.9 puts weight on the current gradient in the actual step. signSGD-momentum signs a single EMA and cannot separate these; NAdam mixes moment and gradient but does not decouple the tracking rate from the application.
- *Memory.* One buffer vs Adam's two; no ε, no √v, no factorization hyperparameters.

**Code (PyTorch).** Single per-parameter buffer `exp_avg` (the momentum); the interpolated momentum is formed transiently and signed in place.

```python
from __future__ import annotations
from typing import Tuple, Callable
import torch
from torch.optim.optimizer import Optimizer


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # decoupled weight decay
    p.data.mul_(1. - lr * wd)
    # signed step on the interpolated momentum
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1. - beta1).sign_()
    p.add_(update, alpha=-lr)
    # momentum EMA on its own (slower) constant
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
