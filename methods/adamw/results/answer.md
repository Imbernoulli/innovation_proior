# Decoupled Weight Decay (AdamW / SGDW)

## Problem

Deep-learning libraries implement "weight decay" as L2 regularization — they add `λ'·θ` to the gradient. For plain SGD the two are equivalent (after rescaling), so this is harmless. For adaptive gradient methods such as Adam it is *not* equivalent, and the difference is harmful: routing the decay through the gradient makes it inherit Adam's per-coordinate `1/√v̂` scaling, so weights with historically large gradients get regularized least. Adam ends up effectively under-regularized, which is a concrete cause of its worse generalization relative to SGD with momentum on tasks where regularization matters.

## Key idea

Stop folding the decay into the gradient. Apply the original multiplicative weight decay (Hanson & Pratt, 1988) — shrink the weights by `(1 − λ)` in the derivation's per-step notation — as a **separate step, outside the adaptive update**. This:

1. Restores true (uniform) weight decay for adaptive methods.
2. Decouples the regularization strength `λ` from the learning rate `α`, making them independent to tune.
3. In the fixed-preconditioner limit `M = diag(s)⁻¹`, equals L2 on the scale-adjusted penalty `(λ'/2)‖θ ⊙ √s‖²` with `λ' = λ/α` — i.e. the norm scales coordinate `i` by `√sᵢ`, so the squared penalty weight is `sᵢ`, leaning harder on high-gradient (brittle) directions.
4. Sits in the data-independent state-transition prior `N((I − λI)θ, Q)` under the Bayesian-filtering view of adaptive methods (Aitchison, 2018) — the structurally correct place for a shrink term, untouched by per-coordinate uncertainty.

## Why L2 ≠ weight decay for adaptive methods

For an optimizer with preconditioner `M_t` (`M_t ≠ k·I`):

- Decoupled weight decay: `θ_{t+1} = (1 − λ)θ_t − α M_t ∇f_t`.
- L2 with coefficient `λ'`: `θ_{t+1} = θ_t − αλ' M_t θ_t − α M_t ∇f_t`.

Equality for all `θ_t` requires `λθ_t = αλ' M_t θ_t`, i.e. `M_t = (λ/αλ')·I`. Impossible unless the preconditioner is a scalar multiple of the identity. For plain SGD, `M_t = I`, so they coincide with `λ' = λ/α`; for Adam they cannot.

## Final algorithm (AdamW)

Given `α=0.001, β₁=0.9, β₂=0.999, ε=1e-8`, algebraic per-step decay `λ`, and schedule multiplier `η_t`; per step on raw loss gradient `g_t = ∇f_t(θ_{t−1})`:

```
m_t = β₁ m_{t−1} + (1 − β₁) g_t
v_t = β₂ v_{t−1} + (1 − β₂) g_t²
m̂_t = m_t / (1 − β₁ᵗ),   v̂_t = v_t / (1 − β₂ᵗ)
θ_t = θ_{t−1} − η_t ( α · m̂_t / (√v̂_t + ε) + λ · θ_{t−1} )
```

The decay term `λ·θ_{t−1}` is separate from the adaptive term and is **not** divided by `√v̂_t`. SGDW is the analogous SGD-with-momentum variant: decay the weights in the parameter-update line rather than injecting `λ'θ` into the momentum buffer.

PyTorch names the code-level coefficient `weight_decay`; its AdamW implementation applies the shrink as `param.mul_(1 - lr * weight_decay)`. Under the notation above, `weight_decay = λ/α` for a fixed base learning rate, or the schedule multiplier is included through the effective learning rate.

**Normalized weight decay.** The optimal `λ` shrinks as the training budget grows (cumulative shrink `≈ (1−λ)^{#updates}`, `#updates = (B/b)·T`). Reparameterize `λ = λ_norm · √(b / (B·T))` (batch `b`, dataset size `B`, epochs `T`) and tune `λ_norm` ("the decay for a single batch pass"); it transfers across budgets and datasets.

**AdamWR.** AdamW with cosine annealing `η_t = 0.5 + 0.5·cos(π T_cur/T_i)` and warm restarts (reset `T_cur→0`, `T_i→T_i·T_mult`), using normalized weight decay with `T` = epochs in the current restart. Warm restarts had failed for Adam before because the broken L2 left it under-regularized; with decay fixed, the SGDR schedule carries over.

## Code

Faithful to the canonical PyTorch implementation (`torch/optim/adamw.py`): the decoupled decay is the single line `param.mul_(1 - lr * weight_decay)`, applied before the Adam moment update, and the gradient used in the moments is the raw loss gradient.

```python
import math
import torch


class AdamW:
    """Adam with decoupled weight decay.

    The decay is a direct (1 - lr*wd) multiply on the parameters, separate from
    the gradient-based Adam step -- the gradient feeding the moments is the raw
    loss gradient (no wd*param added to it).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, schedule=None):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.schedule = schedule or (lambda step: 1.0)   # eta_t hook
        self.state = {id(p): dict(step=0,
                                  m=torch.zeros_like(p),
                                  v=torch.zeros_like(p))
                      for p in self.params}

    @torch.no_grad()
    def step(self):
        b1, b2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[id(p)]
            st["step"] += 1
            eta = self.schedule(st["step"])

            # decoupled weight decay: applied to the weights, not the gradient
            p.mul_(1 - eta * self.lr * self.wd)

            # Adam moments on the raw gradient
            st["m"].mul_(b1).add_(g, alpha=1 - b1)
            st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)

            bc1 = 1 - b1 ** st["step"]
            bc2 = 1 - b2 ** st["step"]
            denom = (st["v"] / bc2).sqrt().add_(self.eps)
            step_size = eta * self.lr / bc1
            p.addcdiv_(st["m"], denom, value=-step_size)

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class SGDW:
    """SGD with momentum and decoupled weight decay."""

    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-4, schedule=None):
        self.params = list(params)
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.schedule = schedule or (lambda step: 1.0)
        self.state = {id(p): dict(step=0, m=torch.zeros_like(p)) for p in self.params}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad                              # raw gradient; wd kept out of the buffer
            st = self.state[id(p)]
            st["step"] += 1
            eta = self.schedule(st["step"])
            p.mul_(1 - eta * self.lr * self.wd)     # direct shrink of the old weights
            st["m"].mul_(self.momentum).add_(g, alpha=eta * self.lr)
            p.add_(st["m"], alpha=-1)               # momentum/gradient step

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


def normalized_weight_decay(lambda_norm, batch_size, dataset_size, epochs):
    """Algebraic per-step lambda = lambda_norm * sqrt(b / (B*T))."""
    return lambda_norm * math.sqrt(batch_size / (dataset_size * epochs))


def pytorch_weight_decay(lambda_norm, batch_size, dataset_size, epochs, lr):
    """Coefficient to pass as PyTorch AdamW weight_decay for the normalized lambda."""
    return normalized_weight_decay(lambda_norm, batch_size, dataset_size, epochs) / lr


def cosine_eta(T_cur, T_i, eta_min=0.0, eta_max=1.0):
    """AdamWR/SGDR multiplier; a warm restart resets T_cur->0 and T_i->T_i*T_mult."""
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * T_cur / T_i))
```

The only change from standard Adam is the decoupling: `grad` is never modified by the weight decay, and the decay appears as a direct multiply on the parameters.
