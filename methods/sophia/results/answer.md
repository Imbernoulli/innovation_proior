# Sophia: Second-order Clipped Stochastic Optimization

## Problem

Language-model pre-training cost is roughly (number of optimizer steps) × (cost per step). Adam has been the default for years. The goal is to reach the same validation loss in far fewer steps **without** raising the average per-step cost, so the step-count saving converts into less compute and wall-clock time. The obstacle is that LM loss landscapes have curvature spread over orders of magnitude across coordinates (heterogeneous curvature), are non-convex (indefinite Hessian), and change quickly along the trajectory — and the classical curvature-aware methods that would exploit this are too expensive per step to win on the clock.

## Key idea

Precondition the gradient by a cheap, infrequently-refreshed estimate of the **positive diagonal of the Hessian**, and bound the worst-case update with a **per-coordinate clip**.

- Adam ≈ SignGD takes a uniform-magnitude step in every coordinate, so it equalizes step size, not loss decrease; on heterogeneous curvature the flat directions starve and the rate pays the condition number.
- The curvature is exactly the quantity that equalizes loss decrease per coordinate (Newton: step g/h empties each local quadratic), and Newton's rate is condition-number-free.
- The full Hessian is infeasible (size), unsafe (indefinite → climbs to saddles/maxima), and unreliable (non-stationary). Sophia fixes all three: use only the **diagonal**, only **positive** curvature, refreshed only every **k** steps, with a **clip** that caps the damage of any bad coordinate.
- The clip makes the optimizer fall back to a bounded SignGD step wherever curvature is negative, tiny, stale, or mis-estimated, and take the full curvature-scaled step where it is trustworthy. This bounded-decrease safeguard is what licenses estimating the Hessian only every k≈10 steps, keeping the overhead ~5%.

## Algorithm

Maintain an EMA of the gradient (numerator) and an EMA of the diagonal-Hessian estimate (denominator):

- m_t = β₁ m_{t−1} + (1 − β₁) g_t
- every k steps: ĥ_t = Estimator(θ_t); h_t = β₂ h_{t−k} + (1 − β₂) ĥ_t; otherwise h_t = h_{t−1}
- θ_t ← θ_t − η_t λ θ_t                         (decoupled weight decay)
- θ_{t+1} = θ_t − η_t · clip( m_t / max(γ h_t, ε), 1 )   (all operations element-wise)

with clip(z, ρ) = max(min(z, ρ), −ρ), small ε > 0 (e.g. 1e-12), and γ controlling the clipped fraction. The identity η·clip(m/max(γh,ε),1) = (η/γ)·clip(m/max(h,ε/γ),γ) shows γ sets the clip threshold on the raw ratio m/h while every clipped coordinate contributes exactly η, so the update scale is decoupled from γ; γ is tuned to keep most coordinates clipped (≈50–90%). Where h_i ≤ 0 the floor ε forces the update to η·sign(m_i) — momentum SignSGD as an automatic backup.

### Diagonal Hessian estimators

**(1) Hutchinson (unbiased, structure-agnostic, needs a Hessian-vector product).** Draw u ~ 𝒩(0, I_d); return ĥ = u ⊙ (∇²L(θ) u). Unbiased because E[u_i (Hu)_i] = Σ_j H_{ij} E[u_i u_j] = H_{ii}. The product Hu = ∇⟨∇L, u⟩ is one double-backward; the full Hessian is never formed. Entries may be negative (handled by the floor/clip).

**(2) Gauss-Newton-Bartlett (biased, always PSD, gradient-only).** For ℓ = ce(f(θ,x), y) with logits f ∈ ℝ^V, the Hessian splits as ∇²ℓ = J_θf · S · J_θfᵀ + J_θθf[q]; drop the small second term and estimate the Gauss-Newton term J_θf S J_θfᵀ (PSD ⇒ descent direction). Since S = diag(p) − ppᵀ (p = softmax(f)) depends only on the logits, not the label, and ce against a model-sampled label is the categorical NLL, Bartlett's second identity gives S = E_{ŷ~Cat(f)}[(∂ce/∂t)(∂ce/∂t)ᵀ], hence diag(J S Jᵀ) = E_{ŷ~Cat(f)}[∇_θ ce(f,ŷ) ⊙ ∇_θ ce(f,ŷ)]. Autodiff only exposes the averaged minibatch gradient, but Bartlett's first identity (E_{ŷ}[∇ce] = 0) kills the cross terms, so for L̂ = (1/B)Σ_b ce(f(θ,x_b), ŷ_b) on sampled labels, E[B · ∇L̂ ⊙ ∇L̂] = diag of the minibatch Gauss-Newton matrix. Estimator: sample labels from the model, take one ordinary gradient, square it element-wise, scale by B. Always non-negative; cost ≈ one gradient.

Sophia-H uses estimator (1); Sophia-G uses estimator (2). Both cost on the order of one gradient and add ~5% per-step overhead at k=10.

## Why it works (convex analysis)

For the deterministic clipped-Newton iterate θ_+ = θ − η Vᵀ clip(V(∇²L)⁻¹∇L, ρ) (eigendecomposition ∇²L = VᵀΣV), under strict convexity and a multiplicative Hessian-Lipschitz condition, the **descent lemma** gives

  L(θ_+) − L(θ) ≤ −(η − η²) Σ_i min{ ρ|v_iᵀ∇L|, σ_i⁻¹|v_iᵀ∇L|² }.

Each coordinate is guaranteed the smaller of the full Newton decrease σ_i⁻¹|v_iᵀ∇L|² (note: no penalty for large σ_i) and the safe clipped decrease ρ|v_iᵀ∇L|. A burn-in phase reaches a neighborhood of the minimum; thereafter the clip never triggers and the iterate becomes pure Newton with exponential decay. With η = 1/2, ρ = R/(2√d):

  T ≲ d · (L(θ₀) − minL)/(μR²) + ln( μR²/(32dε) ),

independent of the condition number and the smoothness (largest-eigenvalue) constant. In contrast, SignGD (Adam's proxy) on L = ½μθ₁² + ½βθ₂² needs T ≥ ½(√(Δ/ε) − √2)·√(β/μ) steps — it provably pays the square root of the condition number.

## Code

Grounded in a nanoGPT-style training loop; this is the Sophia-G (GNB) variant.

```python
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F


class SophiaG(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta: {betas}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho: {rho}")
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self):
        # GNB estimate: called after backward() on SAMPLED labels, so p.grad is
        # the gradient of the resampled-label loss. h <- EMA(grad ⊙ grad); always >= 0.
        for group in self.param_groups:
            _, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr, rho, wd = group['lr'], group['rho'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('SophiaG does not support sparse gradients')
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['step'] = torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                m, h = state['exp_avg'], state['hessian']
                state['step'] += 1

                # decoupled weight decay
                p.mul_(1 - lr * wd)
                # gradient EMA (numerator)
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                # theta -= lr * clip(m / max(gamma * h, eps), 1);  gamma == rho, B folded into denom.
                # sign(m) * clip(|m| / d, 1) == clip(m / d, 1) since clip preserves sign.
                ratio = (m.abs() / (rho * bs * h + 1e-15)).clamp(max=1.0)
                p.addcmul_(m.sign(), ratio, value=-lr)
        return loss


# ---- nanoGPT-style training loop ----
# k = hess_interval (e.g. 10); total_bs * block_size = number of tokens per Hessian batch (B)
optimizer = SophiaG(model.parameters(), lr=peak_lr, betas=(0.96, 0.99),
                    rho=0.05, weight_decay=0.2)

for it in range(max_iters):
    # ----- ordinary step: real-label loss, clipped curvature-scaled update -----
    logits, loss = model(X, Y)
    X, Y = get_batch('train')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # global-norm clip (stability)
    optimizer.step(bs=total_bs * block_size)
    optimizer.zero_grad(set_to_none=True)

    # ----- every k steps: refresh GNB curvature on SAMPLED labels (~one extra gradient) -----
    if it % k == k - 1:
        logits, _ = model(X, 0)
        X, Y = get_batch('train')
        y_sample = torch.distributions.Categorical(logits=logits).sample()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               y_sample.view(-1), ignore_index=-1)
        loss.backward()
        optimizer.update_hessian()
        optimizer.zero_grad(set_to_none=True)
```

For Sophia-H, replace the curvature refresh: draw u ~ 𝒩(0, I), compute the Hessian-vector product Hu = ∇⟨∇L, u⟩ (a double backward), and feed u ⊙ (Hu) into the same `hessian` buffer (EMA), instead of resampling labels and squaring the gradient.

Defaults that work well: β₁ = 0.96, β₂ = 0.99, ε = 1e-12, k = 10; γ (= `rho`) ≈ 0.01 for Sophia-H and ≈ 0.05 for Sophia-G, tuned so the unclipped fraction is roughly 10–50% (clipped fraction ~50–90%); decoupled weight decay ≈ 0.2; peak learning rate tuned per model size.
