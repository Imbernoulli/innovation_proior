# LARS — Layer-wise Adaptive Rate Scaling

## Problem

Data-parallel synchronous SGD speeds up convolutional-network training by splitting a large global batch `B` across many workers. But growing `B` (at fixed epochs) means fewer weight updates, so the learning rate must grow with `B` — and the standard recipe (linear LR scaling + warmup) diverges past a batch-size ceiling (~`2K` for AlexNet on ImageNet), and where it survives it loses accuracy. The residual loss is not a generalization gap (the train–test loss gap does not widen at large batch) — it is **under-optimization**.

## Key idea

Measure the per-layer ratio `‖w^ℓ‖ / ‖∇L(w^ℓ)‖`. It spans **orders of magnitude** across layers (≈single digits for an early conv layer, >1000 for a late FC layer). A single global learning rate `λ` is therefore wrong for almost every layer at once: it must stay below the smallest ratio to avoid the worst layer moving more than its own weight norm in one step (divergence), which starves every layer with a larger ratio.

Fix it by giving each layer ℓ its own **local learning rate** equal to a trust coefficient times that ratio, with a single **global** rate `γ` on top:

```
Δw^ℓ = γ · λ^ℓ · ∇L(w^ℓ),     λ^ℓ = η · ‖w^ℓ‖ / ‖∇L(w^ℓ)‖,   η < 1
```

The gradient norm then cancels in the step size:

```
‖Δw^ℓ‖ = γ · η · ‖w^ℓ‖
```

so every layer moves the **same fixed fraction** `γ·η` of its own weight norm, independent of its gradient magnitude. This reduces sensitivity to vanishing or exploding gradient scales, and the worst layer no longer caps the whole network. With weight decay `β`, use the conservative bound `‖g + βw‖ ≤ ‖g‖ + β‖w‖` in the denominator so the decayed direction cannot exceed the trusted relative step:

```
λ^ℓ = η · ‖w^ℓ‖ / ( ‖∇L(w^ℓ)‖ + β·‖w^ℓ‖ )
```

This is distinct from Adam/RMSProp: it adapts **per layer** (not per weight — a more stable aggregate) and controls the step **relative to the weight norm** (not by gradient statistics). It is a special case of block-diagonal rescaling with one scalar block per layer. Warmup remains useful as a smoother for the earliest steps, but the cross-layer learning-rate mismatch is handled inside the optimizer.

## Algorithm (SGD with LARS, momentum + weight decay + poly decay)

```
Params: base LR γ0, momentum m, weight decay β, trust coeff η, steps T
Init:   t = 0, v = 0, init w0^ℓ for each layer ℓ
while t < T, for each layer ℓ:
    g_t^ℓ      ← ∇L(w_t^ℓ)
    γ_t        ← γ0 · (1 − t/T)^2
    r_t^ℓ      ← η · ‖w_t^ℓ‖ / ( ‖g_t^ℓ‖ + β·‖w_t^ℓ‖ + ε )
    u_t^ℓ      ← r_t^ℓ · ( g_t^ℓ + β·w_t^ℓ )
    v_{t+1}^ℓ  ← m·v_t^ℓ + u_t^ℓ              # PyTorch-style momentum buffer
    w_{t+1}^ℓ  ← w_t^ℓ − γ_t·v_{t+1}^ℓ        # global schedule applied once
```

For zero weight norm or zero gradient norm, skip adaptation for that tensor and use multiplier `1.0`.

## Code

The implementation below mirrors the standard PyTorch LARS/LARC optimizer shape: per parameter tensor, compute the trust ratio, fold in weight decay, scale the gradient, then run ordinary momentum SGD with the **global** learning rate. By default it applies adaptation to parameter groups with nonzero weight decay; set `always_adapt=True` to adapt no-decay groups too.

```python
import torch
from torch.optim.optimizer import Optimizer

class Lars(Optimizer):
    """Layer-wise Adaptive Rate Scaling (LARS) on top of momentum SGD.

    For each parameter tensor (treated as a layer), scale its step by the
    trust ratio  eta * ||w|| / (||g|| + wd*||w||)  so the step is a fixed
    fraction of the weight norm, then apply momentum SGD with the global lr.
    Optional LARC `trust_clip` caps the local lr at the global lr.
    """
    def __init__(self, params, lr=1.0, momentum=0, dampening=0,
                 weight_decay=0.0, nesterov=False,
                 trust_coeff=0.001, eps=1e-8, trust_clip=False, always_adapt=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum and zero dampening")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        trust_coeff=trust_coeff, eps=eps,
                        trust_clip=trust_clip, always_adapt=always_adapt)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']     # beta
            momentum     = group['momentum']         # m
            dampening    = group['dampening']
            nesterov     = group['nesterov']
            trust_coeff  = group['trust_coeff']      # eta
            eps          = group['eps']
            lr           = group['lr']               # global gamma_t (set by scheduler)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # --- LARS per-layer trust ratio ---------------------------
                if weight_decay != 0 or group['always_adapt']:
                    w_norm = p.norm(2.0)
                    g_norm = grad.norm(2.0)
                    trust_ratio = trust_coeff * w_norm / (g_norm + w_norm * weight_decay + eps)
                    # no adaptation when either norm is zero
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, trust_ratio, 1.0),
                        1.0,
                    )
                    if group['trust_clip']:
                        trust_ratio = torch.clamp(trust_ratio / lr, max=1.0)
                    grad.add_(p, alpha=weight_decay)         # g + beta*w
                    grad.mul_(trust_ratio)                   # scale by trust ratio

                # --- momentum SGD with the GLOBAL learning rate -----------
                if momentum != 0:
                    state = self.state[p]
                    buf = state.get('momentum_buffer')
                    if buf is None:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)
                    grad = grad.add(buf, alpha=momentum) if nesterov else buf

                p.add_(grad, alpha=-lr)
        return loss
```

Usage: tune the global `lr` (the `γ` schedule, with warmup + polynomial decay) per batch size, keep `trust_coeff ≈ 1e-3`, `momentum=0.9`, `weight_decay` as for the baseline. The per-layer trust ratio removes the cross-layer learning-rate mismatch that otherwise forces the global rate down to whatever the worst-conditioned layer can survive.
