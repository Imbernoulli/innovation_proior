# LAMB: Layerwise-Adaptive Moments for Batch training

## Problem

Scaling deep-network training across many accelerators means scaling the mini-batch size, which (at fixed epochs) cuts the number of optimizer steps in proportion to the batch, forcing a larger learning rate. A single global learning rate is the wrong object at large batch: layers differ by orders of magnitude in the ratio of their weight norm `‖x^(i)‖` to their update norm, so any global rate that makes one layer progress overshoots another and destabilizes training. LAMB makes the per-layer effective step proportional to each layer's own weight norm, decoupling the global learning rate from per-layer geometry, so the batch can scale to tens of thousands of examples with one learning-rate recipe: about 100 minutes for BERT at 32k batch, and about 76 minutes with a 64k/32k mixed-batch schedule.

## Key idea

A two-modification strategy over any base optimizer `A` producing a layerwise update `u_t`: (1) normalize each layer's update to unit `ℓ2` norm; (2) scale the per-layer learning rate by `φ(‖x^(i)‖)`, where `φ(z) = min(max(z, γ_l), γ_u)` is a clipped identity. With `φ = id` the per-layer multiplier `‖x^(i)‖/‖u^(i)‖` reads as an estimate of the inverse local smoothness `1/L_i`. Taking the base `A = Adam` gives LAMB: **two-fold adaptivity** — per-dimension (Adam's `1/√v̂`) inside a layer, and per-layer (the trust ratio) across layers. (Base `A = momentum` gives the trust-ratio scheme LARS, which lacks the per-coordinate adaptivity attention models need and diverges on them at large batch.)

## Algorithm

Inputs: `x_1`, learning rate `{η_t}`, `0 < β₁, β₂ < 1`, scaling function `φ`, `ε > 0`, weight decay `λ`. Set `m_0 = 0`, `v_0 = 0`. For `t = 1..T`:

- `g_t = (1/|S_t|) Σ_{s∈S_t} ∇ℓ(x_t, s)`
- `m_t = β₁ m_{t-1} + (1−β₁) g_t`
- `v_t = β₂ v_{t-1} + (1−β₂) g_t²`
- `m̂_t = m_t/(1−β₁^t)`, `v̂_t = v_t/(1−β₂^t)`   (the algorithm includes this Adam correction; it can be removed when an explicit warmup schedule is already used, as the Google implementation does)
- `r_t = m̂_t/(√v̂_t + ε)`
- for each layer `i`: `x_{t+1}^(i) = x_t^(i) − η_t · φ(‖x_t^(i)‖) / ‖r_t^(i) + λ x_t^(i)‖ · (r_t^(i) + λ x_t^(i))`

The decoupled weight decay `λ x` sits *inside* the trust-ratio numerator and norm so the decay scale stays commensurate with the layer's step. Special cases: `β₁ = β₂ = 0` reduces to signSGD scaled by `√(layer dim)`. Defaults: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-6`, `λ = 0.01`. Across batch sizes, use `√b` learning-rate scaling and linear-epoch warmup; `ℓ2` is the default layer norm (other norms change accuracy by <0.1%).

## Convergence (nonconvex, `b = T`, constant `η`, `α_l ≤ φ ≤ α_u`)

For the LARS/trust-ratio scheme with `β₁ = λ = 0`: `(E[(1/√h)Σ_i ‖∇_i f(x_a)‖])² ≤ O((f(x_1)−f*) L_avg/T + ‖σ‖_1²/(Th))`. For LAMB with `β₁ = λ = 0`, equal layer sizes `d_i = d/h`, and `β₂ = 0`: `(E[(1/√d)‖∇f(x_a)‖_1])² ≤ O((f(x_1)−f*) L_avg/T + ‖σ̃‖_1²/(Th))`. For LAMB with `β₂ > 0`: `E‖∇f(x_a)‖² ≤ O(√(G²d/(h(1−β₂))) · [√(2(f(x_1)−f*)‖L‖_1/T) + ‖σ̃‖_1/√T])`. The `L_avg = (1/h)Σ_i L_i` replacement is the clean win in the LARS and `β₂=0` LAMB cases; the `β₂>0` LAMB theorem is looser. Via a signSGD-style density comparison, the layerwise rate beats SGD when the gradient is denser than the curvature and the noise (`ψ_L ≪ ψ_g²`, `ψ_σ ≪ ψ_g²`).

## Code

```python
import torch
from torch.optim import Optimizer

class Lamb(Optimizer):
    """LAMB: layerwise-adaptive Adam for large-batch training."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, debias=False, adam=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.debias = debias            # TFA bias correction; Google BERT code omits it
        self.adam = adam                # trust_ratio := 1, for ablation
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                if g.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients.')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)      # m
                    state['exp_avg_sq'] = torch.zeros_like(p.data)   # v
                m, v = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Adam moment EMAs
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                step_size = group['lr']
                if self.debias:
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    r = (m / bc1) / ((v / bc2).sqrt().add(group['eps']))
                else:
                    r = m / v.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    r = r.add(p.data, alpha=group['weight_decay'])

                # Canonical Google/TFA core uses phi(z)=z; some PyTorch variants clamp z.
                weight_norm = p.data.pow(2).sum().sqrt()
                r_norm = r.pow(2).sum().sqrt()
                if weight_norm == 0 or r_norm == 0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = weight_norm / r_norm
                if self.adam:
                    trust_ratio = 1.0

                # per-layer step proportional to weight norm; global lr decoupled
                p.data.add_(r, alpha=-step_size * trust_ratio)
        return loss
```
