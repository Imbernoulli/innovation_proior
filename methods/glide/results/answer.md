# GLIDE

## Problem

Build a photorealistic text-to-image diffusion model and a text-driven editor. The core problems are how to inject a free-form caption into a strong image diffusion backbone, and how to guide sampling toward the caption without collapsing diversity.

## Key Idea

Use an ADM-style `ε`-prediction diffusion model with learned variance, condition it on text through both a global caption embedding and token-level attention context, and steer sampling with classifier-free guidance. The score/noise relation is `∇_{x_t} log p_t(x_t|c) ≈ -ε_θ(x_t,t|c)/σ_t`, so the implicit classifier
`p^i(c|x_t) ∝ p(x_t|c)/p(x_t)` has score
`∇_{x_t} log p^i(c|x_t) = score_c - score_∅ ≈ -(ε_c - ε_∅)/σ_t`.
Extrapolating the score toward the caption and converting back to `ε` gives the implementation formula:

`ε̂_θ(x_t|c) = ε_θ(x_t|∅) + s·(ε_θ(x_t|c) - ε_θ(x_t|∅))`, `s ≥ 1`.

CLIP guidance is the comparable external-gradient alternative:
`μ̂_θ(x_t|c) = μ_θ(x_t|c) + s·Σ_θ(x_t|c)∇_{x_t}(f(x_t,t)·g(c))`, using a noised CLIP image encoder so the guidance model sees the same noisy intermediates as the sampler.

## Components

- **Diffusion backbone:** `L_simple = E[‖ε - ε_θ(x_t,t)‖²]`, learned variance, `6` output channels for RGB (`3` noise + `3` variance).
- **Text conditioning:** Transformer over caption tokens; final token replaces the ADM class embedding, and the full token sequence is projected into each attention layer's context.
- **Classifier-free fine-tune:** replace text tokens with the empty sequence `20%` of the time, giving conditional and unconditional predictions from one model.
- **Upsampling:** separate text-conditional diffusion upsampler from `64×64` to `256×256`, conditioned on the low-resolution image.
- **Editing:** inpainting fine-tune with four extra input channels: clean known RGB plus mask, zero-initialized at the input layer; the upsampler receives the full low-resolution image and only the unmasked high-resolution context.

Common large-model configuration: base `64×64` model with a 512-channel visual backbone plus a 24-block, width-2048 text Transformer; upsampler with a width-1024 text encoder; 16-bit training with loss scaling; base sampling uses `150` steps for samples, `100` for inpainting, and `250` for evaluations; fast upsampling uses `27` steps split as `10,10,3,2,2`; classifier-free guidance scale `3.0`, CLIP guidance scale `2.0`.

## Code

```python
import torch as th

def build_text_kwargs(model, prompt, batch_size, text_ctx, device, include_null=False):
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(tokens, text_ctx)

    if not include_null:
        return dict(
            tokens=th.tensor([tokens] * batch_size, device=device),
            mask=th.tensor([mask] * batch_size, dtype=th.bool, device=device),
        )

    null_tokens, null_mask = model.tokenizer.padded_tokens_and_mask([], text_ctx)
    return dict(
        tokens=th.tensor(
            [tokens] * batch_size + [null_tokens] * batch_size,
            device=device,
        ),
        mask=th.tensor(
            [mask] * batch_size + [null_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )

def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)

full_batch_size = batch_size * 2
model_kwargs = build_text_kwargs(
    model, prompt, batch_size, options["text_ctx"], device, include_null=True
)

model.del_cache()
samples = diffusion.p_sample_loop(
    model_fn,
    (full_batch_size, 3, options["image_size"], options["image_size"]),
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=model_kwargs,
    cond_fn=None,
)[:batch_size]
model.del_cache()

with th.no_grad():
    z_t = clip_model.text_embeddings([prompt] * batch_size)

def cond_fn(x, t, grad_scale=clip_guidance_scale, **kwargs):
    with th.enable_grad():
        x_var = x.detach().requires_grad_(True)
        z_i = clip_model.image_embeddings(x_var, t)
        loss = th.exp(clip_model.logit_scale) * (z_t * z_i).sum()
        grad = th.autograd.grad(loss, x_var)[0].detach()
    return grad * grad_scale

clip_guided = diffusion.p_sample_loop(
    model,
    (batch_size, 3, options["image_size"], options["image_size"]),
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=build_text_kwargs(
        model, prompt, batch_size, options["text_ctx"], device
    ),
    cond_fn=cond_fn,
)

def build_upsample_kwargs(model_up, prompt, low_res, batch_size, text_ctx, device):
    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(tokens, text_ctx)
    return dict(
        low_res=((low_res + 1) * 127.5).round() / 127.5 - 1,
        tokens=th.tensor([tokens] * batch_size, device=device),
        mask=th.tensor([mask] * batch_size, dtype=th.bool, device=device),
    )

upsample_kwargs = build_upsample_kwargs(
    model_up, prompt, samples, batch_size, options_up["text_ctx"], device
)
up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])

model_up.del_cache()
up_samples = diffusion_up.ddim_sample_loop(
    model_up,
    up_shape,
    noise=th.randn(up_shape, device=device) * upsample_temp,
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=upsample_kwargs,
    cond_fn=None,
)[:batch_size]
model_up.del_cache()

model_kwargs.update(
    inpaint_image=(source_image_64 * source_mask_64)
    .repeat(full_batch_size, 1, 1, 1)
    .to(device),
    inpaint_mask=source_mask_64.repeat(full_batch_size, 1, 1, 1).to(device),
)

def denoised_fn(x_start):
    return (
        x_start * (1 - model_kwargs["inpaint_mask"])
        + model_kwargs["inpaint_image"] * model_kwargs["inpaint_mask"]
    )

edited = diffusion.p_sample_loop(
    model_fn,
    (full_batch_size, 3, options["image_size"], options["image_size"]),
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=model_kwargs,
    cond_fn=None,
    denoised_fn=denoised_fn,
)[:batch_size]
```
