# Classifier-Free Guidance

## Problem

A class-conditional diffusion model samples `x ~ p(x | c)` with good diversity but uneven per-sample fidelity. GANs and flows have a sampling-time knob (truncation / low temperature) that trades diversity for fidelity; diffusion's naive analogues (scaling the score, shrinking the reverse-process noise) only blur samples. Classifier guidance (Dhariwal & Nichol 2021) supplies such a knob but requires a separate classifier trained on noisy images, and its sampling direction is literally a classifier gradient (an adversarial-attack-shaped step). **Classifier-free guidance** obtains the same fidelity↔diversity tradeoff with no classifier at all.

## Key idea

We want to sample the sharpened distribution
```
p̃(z_λ | c) ∝ p(z_λ | c) · p(c | z_λ)^w .
```
Its score is `∇ log p̃(z_λ|c) = ∇ log p(z_λ|c) + w ∇ log p(c|z_λ)`. By Bayes, `p(c|z) = p(z|c)p(c)/p(z)`, so the label prior drops under the gradient and
```
∇_{z_λ} log p(c | z_λ) = ∇_{z_λ} log p(z_λ | c) − ∇_{z_λ} log p(z_λ).
```
This **implicit classifier** is just the conditional minus the unconditional generative score — no trained classifier. Substituting,
```
∇ log p̃(z_λ|c) = (1 + w) ∇ log p(z_λ|c) − w ∇ log p(z_λ).
```
A diffusion model's noise prediction is the score up to a known scale, `ε(z_λ) = −σ_λ ∇ log p(z_λ)` (denoising score matching). Multiplying the combined score by `−σ_λ` gives the **classifier-free guided noise estimate**:
```
ε̃(z_λ, c) = (1 + w) ε(z_λ, c) − w ε(z_λ)   =   ε(z_λ) + (1 + w)·(ε(z_λ, c) − ε(z_λ)).
```
`w = 0` is plain conditional sampling; `w > 0` extrapolates past the conditional prediction, away from the unconditional one. Equivalently `p̃(z|c) ∝ p(z|c)^{1+w}/p(z)^w`: raise the conditional likelihood and **lower the unconditional likelihood** (a negative score term), concentrating mass on class-typical, high-confidence samples. Because `ε(z,c)` and `ε(z)` are outputs of unconstrained networks, their difference is generally not the gradient of any scalar potential — `ε̃` is not literally any classifier's gradient, so the step is not an adversarial attack on a classifier.

To obtain both `ε(z_λ, c)` and `ε(z_λ)` from **one** network: designate a null token `∅`, define `ε(z_λ) := ε(z_λ, c=∅)`, and train with **conditioning dropout** — replace `c` with `∅` with probability `p_uncond`. This is one extra line at training and one at sampling, with one shared denoising network rather than a separate classifier or a second diffusion model.

## Algorithm

Training (joint conditional + unconditional via dropout):
1. Sample `(x, c)`, log-SNR `λ ~ p(λ)`, noise `ε ~ N(0,I)`.
2. With probability `p_uncond`, set `c ← ∅`.
3. Form `z_λ = α_λ x + σ_λ ε` and take a gradient step on `‖ε_θ(z_λ, c) − ε‖²`.

Sampling (guidance strength `w`, increasing log-SNR `λ_1 … λ_T`):
1. `z_1 ~ N(0, I)`.
2. For each step: `ε̃ = (1+w) ε_θ(z_t, c) − w ε_θ(z_t, ∅)`; `x̃ = (z_t − σ_{λ_t} ε̃)/α_{λ_t}`; take the ordinary ancestral reverse step using `x̃` (or substitute another sampler such as DDIM). Each step costs two model evaluations (conditional and unconditional).

Sweeping `w` is the sampling-time knob: larger values push harder toward class-typical samples and reduce diversity, so quality and coverage should be evaluated as a curve rather than at one fixed setting.

## Code

Grounded in a standard clean conditional-diffusion implementation; the only method-specific pieces are conditioning dropout (training) and the linear score combination (sampling).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def alpha_sigma(lam):
    # variance-preserving: alpha_lambda^2 = sigmoid(lambda), sigma_lambda^2 = 1 - alpha^2
    alpha_sq = torch.sigmoid(lam)
    return alpha_sq.sqrt(), (1.0 - alpha_sq).sqrt()


def expand_to_data(v, x):
    if v.ndim == 0:
        v = v.expand(x.shape[0])
    return v.view(-1, *([1] * (x.ndim - 1)))


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    return torch.rand(shape, device=device) < prob


class NoisePredictor(nn.Module):
    """ε_θ(z_λ, λ, c). A learned null embedding ∅ lets one network act as both
    the conditional model ε_θ(z, c) and the unconditional model ε_θ(z, ∅)."""

    def __init__(self, num_classes, cond_dim, backbone):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, cond_dim)
        self.null_emb = nn.Parameter(torch.randn(cond_dim))  # the ∅ token
        self.backbone = backbone  # any net consuming (z, lambda, conditioning embedding)

    def forward(self, z, lam, c, cond_drop_prob=0.0):
        cond = self.class_emb(c)
        if cond_drop_prob > 0:
            keep = prob_mask_like((z.shape[0],), 1.0 - cond_drop_prob, z.device)
            null_cond = self.null_emb[None, :].expand_as(cond)
            cond = torch.where(keep[:, None], cond, null_cond)  # drop label -> ∅
        return self.backbone(z, lam, cond)

    @torch.no_grad()
    def forward_with_cond_scale(self, z, lam, c, cond_scale=1.0):
        """Classifier-free estimate with cond_scale = w + 1."""
        eps_c = self.forward(z, lam, c, cond_drop_prob=0.0)          # ε_θ(z, c)
        if cond_scale == 1.0:
            return eps_c
        eps_uncond = self.forward(z, lam, c, cond_drop_prob=1.0)     # ε_θ(z, ∅)
        return eps_c + (eps_c - eps_uncond) * (cond_scale - 1.0)


def diffusion_loss(model, x, c, p_uncond=0.1):
    """Joint training: denoising MSE with conditioning dropout."""
    lam = sample_log_snr(x.shape[0], x.device)           # λ ~ p(λ)
    eps = torch.randn_like(x)
    a, s = alpha_sigma(expand_to_data(lam, x))
    z = a * x + s * eps                                  # z_λ = α_λ x + σ_λ ε
    eps_pred = model(z, lam, c, cond_drop_prob=p_uncond)  # the one-line training change
    return F.mse_loss(eps_pred, eps)


@torch.no_grad()
def guided_eps(model, z, lam, c, w):
    return model.forward_with_cond_scale(z, lam, c, cond_scale=1.0 + w)


def predict_x_from_eps(z, lam, eps_hat):
    a, s = alpha_sigma(expand_to_data(lam, z))
    return (z - s * eps_hat) / a                         # x_θ = (z_λ − σ_λ ε̃)/α_λ


@torch.no_grad()
def sample(model, c, schedule, w, shape, v):
    """Conditional sampling with guidance strength w."""
    z = torch.randn(shape, device=c.device)              # z ~ N(0, I)
    x_pred = None
    for lam, lam_next in schedule:                       # increasing log-SNR
        eps_hat = guided_eps(model, z, lam, c, w)        # ε̃ = (1+w)ε_c − w ε_∅
        x_pred = predict_x_from_eps(z, lam, eps_hat)
        z = reverse_step(z, lam, lam_next, x_pred, v)    # ancestral step (or DDIM)
    return x_pred
```

Convention note: the guidance strength here is `w` (`w=0` ⇒ unguided conditional). Many implementations instead expose a guidance scale `s = w + 1`, with `ε̃ = ε_∅ + s·(ε_c − ε_∅)`; the two are identical.
