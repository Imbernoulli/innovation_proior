# AdaBelief

**Problem.** Adaptive optimizers (Adam) converge fast but generalize worse than SGD on many tasks, while SGD generalizes well but is slow and brittle; adaptive methods are also the default for unstable settings like GANs. The goal is one optimizer with fast convergence, SGD-like generalization, and training stability — as a one-line, no-new-hyperparameter modification of Adam.

**Key idea.** Adam divides the smoothed gradient by √vₜ, where vₜ is the EMA of the *squared gradient* gₜ² — a magnitude. AdaBelief divides instead by √sₜ, where sₜ is the EMA of the *squared prediction error* (gₜ − mₜ)². Since mₜ (the EMA of the gradient) is a running prediction of the gradient, 1/√sₜ is the "belief" in the observed gradient: if gₜ agrees with the prediction mₜ, sₜ is small and the step is large (confident); if gₜ deviates, sₜ is large and the step is small (cautious). Statistically sₜ ≈ Var(gₜ), so it is an adaptive-variance (rather than adaptive-second-moment) method, and it retains the sign information that squaring the raw gradient discards.

**Why the centered denominator helps.** Consider the step magnitude an ideal (curvature-aware) optimizer would take:
- Flat region (small gradient, small curvature): want a large step. Adam and AdaBelief both take it (small denominator).
- Steep oscillating valley (large gradient, large curvature): want a small step. Both take it (large denominator).
- Large gradient but *small* curvature — the gradient is large yet barely changing: want a large step. Adam takes a *small* step (√vₜ large because |g| is large) — wrong; AdaBelief takes a *large* step (gₜ ≈ mₜ ⇒ sₜ small) — right, matching SGD's behavior in the regime where SGD generalizes better.

2D illustration on f(x,y) = |x| + |y| (gradients ±1), marching in x while oscillating in y: Adam gets vₓ = v_y = 1 (squaring drops the sign) so it steps equally in both; AdaBelief gets sₓ ≈ Var(gₓ) = 0 and s_y ≈ Var(g_y) = 1, so it steps large in the consistent direction x and small in the oscillating direction y.

**Algorithm.** Differences from Adam in brackets. α = 1e-3, β₁ = 0.9, β₂ = 0.999, ε = 1e-8 (Adam's defaults; no new hyperparameters).

```
m_t = β₁ m_{t-1} + (1 − β₁) g_t
s_t = β₂ s_{t-1} + (1 − β₂) (g_t − m_t)²  + ε          [Adam: v_t = β₂ v_{t-1} + (1 − β₂) g_t²]
m̂_t = m_t / (1 − β₁ᵗ),   ŝ_t = s_t / (1 − β₂ᵗ)
θ_t = θ_{t-1} − α · m̂_t / (√ŝ_t + ε)                   [Adam divides by √v̂_t]
```

The +ε inside the sₜ update lower-bounds the denominator, keeping the step finite when belief is near-perfect (sₜ → 0). Optional, orthogonal add-ons: AMSGrad-style running max of sₜ; RAdam-style rectification of the early-step variance; AdamW-style decoupled weight decay.

**Curvature link.** A finite-difference Hessian diagonal is Hᵢᵢ ≈ [gᵢ(θ+δ) − gᵢ(θ)]/δ; identifying the prediction with mₜ and the observation with gₜ gives √sₜ ≈ |gₜ − mₜ| ∝ the gradient's change, i.e. curvature — so √sₜ is a cheap curvature scaling, whereas √vₜ is only a magnitude.

**Code (PyTorch).**

```python
import math
import torch
from torch.optim.optimizer import Optimizer


class AdaBelief(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, weight_decouple=True, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.weight_decouple = weight_decouple
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
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)       # m_t
                    state['exp_avg_var'] = torch.zeros_like(p.data)   # s_t (~ Var g_t)
                    if group['amsgrad']:
                        state['max_exp_avg_var'] = torch.zeros_like(p.data)

                if self.weight_decouple and group['weight_decay'] != 0:
                    p.data.mul_(1.0 - group['lr'] * group['weight_decay'])

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)        # m_t
                grad_residual = grad - exp_avg                          # g_t - m_t
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)  # s_t

                if group['amsgrad']:
                    max_exp_avg_var = state['max_exp_avg_var']
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)      # θ -= lr * m̂ / (√ŝ + eps)

        return loss
```
