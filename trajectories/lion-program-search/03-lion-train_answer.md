Both single-constant readings beat PowerSign and AddSign cleanly but neither clears AdamW —
78.23/78.19 against AdamW's 78.89 on ViT-S/16, 79.54/79.90 against 80.12 on ViT-B/16. So a single
accumulated-then-signed buffer beats a restricted-tree search but not the incumbent, and I need to
understand what forcing one β to do everything is costing. That single β sits inside `interp(g, m,
β)` and immediately gets signed — it's asked to do two jobs at once: remember gradient history,
which wants β close to 1 for a long memory, and stay reactive to the current gradient before
signing, which wants real weight on the fresh gradient. β=0.9 buys reactivity at the cost of
memory; β=0.99 buys memory at the cost of reactivity. Both ablations, read this way, are the same
tradeoff sampled at two points on one losing dial — which makes a sharp prediction: no single β
should do noticeably better than either, since the dial itself is the bottleneck.

Before spending more runs testing that prediction with more single-β sweeps, I went back to what
the search actually produced. The raw discovered program never used one constant — it chained two,
one buffer updated from the gradient and the previous value of a second buffer at ≈0.9, that
second buffer then updated from the gradient and the just-updated first buffer at ≈1.1 (above one:
an extrapolation, not an interpolation — something I'd thrown away as noise when I collapsed to a
single buffer last rung). Substituting one equation into the other rather than eyeballing it:
writing `m_t = (1−b1)·g_t + b1·v_{t−1}` and `v_t = (1−b2)·g_t + b2·m_t`, plugging in and collecting
terms gives a `g_t` coefficient of `(1−b2) + b2·(1−b1)` and a `v_{t−1}` coefficient of `b2·b1`. At
the rounded constants (b1=0.9, b2=1.1) those are 0.01 and 0.99 — summing to exactly 1, which is
what composing two affine interpolations always does, not a coincidence of these particular
numbers. So `v_t = 0.01·g_t + 0.99·v_{t−1}`, precisely a single EMA at β≈0.99 run directly on the
raw gradient — the second buffer never needed the first buffer's help. The same algebra at the raw
unrounded constants (0.8999999761581421, 1.109133005142212) gives β≈0.99822, so this isn't a
rounding artifact either way I check it.

That means there's really only one buffer worth storing — the slow EMA at β₂≈0.99 — while the
quantity that actually gets signed each step is a *different*, never-stored blend,
`interp(g_t, m_{t−1}, β₁)` with β₁≈0.9, recomputed fresh every step and thrown away right after
signing. Memory lives in β₂, applied to what persists; reactivity lives in β₁, applied only to the
transient signed quantity — exactly the two-dial structure the failure mode predicted, but achieved
by storing one buffer at the slow rate and recomputing the fast blend, not by storing two buffers.
Neither Ablation_0.9 nor Ablation_0.99 could express this, since both forced the persisted and
signed quantities to be identical. Two more pieces close out cleanly: `clip`/`arcsin` on the
incoming gradient show no quality drop when ablated, so I drop them; and a `cosh(update)` statement
assigns into a variable that's fully overwritten at the top of the next iteration before anything
reads it — a directly checkable dead-code fact, not something that needs testing, so I drop it
unconditionally.

I propose the decoupled two-constant rule the algebra actually supports: one persisted buffer `m`
updated on its own slow constant β₂≈0.99, a step formed by signing a fast blend of that buffer with
the current gradient at β₁≈0.9, decoupled weight decay, no ε, no square root, no second full
buffer. The signed step is exactly ±1 per coordinate before weight decay, larger norm than AdamW's
or the ablations' step at the same nominal lr, so I carry the same lr reduction into tuning, and
since effective decoupled weight decay is lr·λ, I let λ move opposite lr rather than changing two
things at once. I don't yet know whether this clears AdamW — the memory/reactivity mechanism
explains why the single-constant readings failed, it doesn't guarantee decoupling recovers the lost
ground. What it does give me is a sharp, falsifiable expectation: if the mechanism is right, this
rule should beat both ablations by a real, consistent margin at both scales, since it's the one
structure on this ladder that gives memory and reactivity separate dials instead of one shared one.

```
cₜ = β₁·mₜ₋₁ + (1−β₁)·gₜ                  # interpolated momentum used for the step
θₜ = θₜ₋₁ − ηₜ·( sign(cₜ) + λ·θₜ₋₁ )       # signed step + decoupled weight decay
mₜ = β₂·mₜ₋₁ + (1−β₂)·gₜ                  # momentum buffer, its own slower EMA
```

```python
from __future__ import annotations
from typing import Tuple, Callable
import torch
from torch.optim.optimizer import Optimizer


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    p.data.mul_(1. - lr * wd)
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1. - beta1).sign_()
    p.add_(update, alpha=-lr)
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
