# GLIDE

## Problem

Turn a diffusion model into a photorealistic **text-to-image** generator and **text-driven editor** (inpainting). Diffusion had become the strongest backbone for class-conditional synthesis; the open questions are how to condition the denoiser on free-form text and — the decisive one for sample quality — which **guidance** strategy to use: CLIP guidance or classifier-free guidance.

## Key idea

Condition an ADM diffusion model on a caption and steer the reverse process with **classifier-free guidance**, which needs no separate classifier. Train the model to denoise both with the caption and unconditionally (caption dropped to the empty sequence part of the time); at sampling, extrapolate the noise prediction along the conditional direction:
`ε̂_θ(x_t|c) = ε_θ(x_t|∅) + s·(ε_θ(x_t|c) − ε_θ(x_t|∅))`, `s ≥ 1`.
This is exactly classifier guidance using the model's own *implicit* classifier `p^i(y|x_t) ∝ p(x_t|y)/p(x_t)`, whose gradient is `∇ log p(x_t|y) − ∇ log p(x_t) ∝ ε*(x_t|y) − ε*(x_t)` (the score equals the noise prediction). The scale `s` trades diversity for fidelity. Classifier-free guidance is favored over CLIP guidance because the model steers with its own full conditional/unconditional knowledge rather than a narrower similarity score, and it needs no extra noised-CLIP encoder.

## Components

- **Diffusion backbone.** `ε`-prediction with the re-weighted surrogate `L_simple = E[‖ε − ε_θ(x_t,t)‖²]`, learned variance `Σ_θ` (fewer sampling steps); score relation `∇_{x_t} log p(x_t) ∝ ε_θ`. The denoiser outputs 6 channels per RGB image: 3 for `ε`, 3 for the variance.
- **Text conditioning.** Encode the caption into `K` tokens with a Transformer; use the final token embedding in place of the ADM class embedding (global conditioning), and project the full `K`-token sequence to each attention layer and concatenate it to that layer's attention context (per-layer cross-attention).
- **Classifier-free training.** After pretraining, fine-tune with the token sequence replaced by the empty sequence `20%` of the time, giving a clean unconditional `ε_θ(·|∅)` alongside the conditional model.
- **CLIP guidance (alternative).** Perturb the reverse mean by the gradient of a **noised-CLIP** similarity: `μ̂_θ(x_t|c) = μ_θ(x_t|c) + s·Σ_θ(x_t|c) ∇_{x_t}(f(x_t)·g(c))`. CLIP must be trained on noised images `f(x_t,t)` (same noise schedule), since the intermediate `x_t` are out-of-distribution for clean-image CLIP.
- **Upsampling.** A separate text-conditional diffusion model upsamples `64×64 → 256×256`, conditioning on the low-res image via channel concatenation (SR3-style).
- **Inpainting/editing.** Fine-tune for the task (Palette-style): add 4 UNet input channels (a second RGB set for the clean known region + a mask channel), zero-initialize their input weights; the model fills the masked region from the clean context plus caption — no edge artifacts.

**Defaults:** base 3.5B params (512-channel ADM UNet ~2.3B + 24-block, width-2048 text Transformer ~1.2B), 64×64, trained 2.5M iters at batch 2048; upsampler 1.5B, batch 512, 1.6M iters; learned variance; 16-bit + loss scaling; DALL-E dataset. Sampling: 150 base steps (100 for inpainting; 250 for evals); upsampler strided schedule of 27 steps (segments `10,10,3,2,2`). Sample guidance scales: classifier-free `3.0`, CLIP `2.0`.

## Code

```python
import torch as th

# --- Classifier-free guidance: one batched forward over [cond ; uncond] halves ---
def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]               # 3 noise + variance channels
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)

# model_kwargs: caption tokens (+ mask) on the conditional half, empty tokens on the
# unconditional half, stacked to the doubled batch.
samples = diffusion.p_sample_loop(
    model_fn,
    (full_batch_size, 3, options["image_size"], options["image_size"]),
    clip_denoised=True,
    model_kwargs=model_kwargs,
    cond_fn=None,
)

# --- CLIP guidance alternative: mean-perturbing cond_fn from a noised CLIP model ---
def cond_fn(x, t, grad_scale=grad_scale, **kwargs):
    with th.enable_grad():
        x_var = x.detach().requires_grad_(True)
        z_i = self.image_embeddings(x_var, t)                    # noised-CLIP image embed f(x_t, t)
        loss = th.exp(self.logit_scale) * (z_t * z_i).sum()      # f(x_t) · g(c); z_t = caption embed
        grad = th.autograd.grad(loss, x_var)[0].detach()
    return grad * grad_scale

samples = diffusion.p_sample_loop(
    model, (batch_size, 3, options["image_size"], options["image_size"]),
    clip_denoised=True, model_kwargs=model_kwargs,
    cond_fn=cond_fn,                                             # μ̂ = μ + s·Σ·∇(f·g)
)
```
