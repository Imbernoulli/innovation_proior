The goal is to turn a diffusion model into a photorealistic text-to-image generator and editor. Class-conditional diffusion models with classifier guidance already produce the strongest unconditional and class-conditional samples, so they are the natural backbone. The challenge is replacing the class label with an arbitrary natural-language caption and steering generation toward that caption without losing diversity or photorealism.

Existing approaches each have a clear weakness. DALL-E generates images autoregressively over discrete VQ-VAE tokens and then reranks many samples with CLIP, which is slow and not a single end-to-end conditional sampler. Text-conditional GANs are fast but struggle with compositional fidelity on open-domain captions. Classifier guidance improves diffusion samples, but it needs a separate classifier trained on noised images, and it is built for a fixed label set rather than free-form sentences. Finally, publicly available CLIP models were trained on clean images, so feeding them the noisy intermediates produced by a diffusion sampler puts them out of distribution; they need hand-engineered augmentations and auxiliary losses to give recognizable gradients. What is needed is a noise-aware guidance signal and a way to inject fine-grained text into the denoiser.

The method is GLIDE. It keeps an ADM-style epsilon-prediction diffusion backbone with learned variance, conditions it on text through both a global caption embedding and token-level attention context, and steers sampling with classifier-free guidance. It also explores a noised-CLIP guidance variant for comparison. The base model generates a 64x64 image from text, a separate text-conditional diffusion upsampler refines it to 256x256, and an inpainting fine-tune turns the same pipeline into a text-driven editor.

Text enters the denoiser at two levels. A Transformer encodes the caption; its final token embedding replaces the class embedding that ADM would use, giving every layer a global sentence-level signal. The full sequence of token embeddings is also projected into the width of each attention block and concatenated to the attention context, so spatial features can bind individual words, attributes, and relations at multiple resolutions. The diffusion loss remains the standard noise regression objective L_simple = E_{t,x_0,epsilon}[||epsilon - epsilon_theta(x_t,t)||^2], with the model outputting six channels per RGB image: three for predicted noise and three for learned variance interpolation.

Guidance comes from classifier-free guidance. During fine-tuning, the text tokens are randomly replaced by the empty sequence about 20% of the time. The same network therefore learns both a conditional prediction epsilon_theta(x_t|c) and an unconditional prediction epsilon_theta(x_t|empty). The implicit classifier p^i(c|x_t) proportional to p(x_t|c)/p(x_t) has score score_c - score_empty, which in the epsilon parameterization becomes epsilon_hat_theta(x_t|c) = epsilon_theta(x_t|empty) + s * (epsilon_theta(x_t|c) - epsilon_theta(x_t|empty)) with guidance scale s >= 1. This extrapolates the noise prediction toward the caption direction without any separate classifier. At sampling time the denoiser is run on a doubled batch, half captioned and half empty, and the two noise predictions are combined before converting back to the reverse mean. A noised CLIP encoder trained on the same noise schedule is also evaluated as an external-gradient alternative, using the sampler's existing cond_fn hook to perturb the mean, but classifier-free guidance is the main mechanism because it uses the model's own learned conditional distribution.

Resolution is handled in two stages. The base model works at 64x64 for semantics and composition; a separate text-conditional super-resolution diffusion model then upsamples to 256x256, conditioning on the low-resolution image by channel concatenation. For editing, the base model is fine-tuned on inpainting by adding four input channels: the clean known RGB pixels and a single-channel mask. Those new input weights are initialized to zero so the model starts from the ordinary text-to-image generator and gradually learns to respect the provided context. During sampling, a projection on the predicted clean image copies known pixels from the source image inside the masked region and leaves the rest to the model. The upsampler receives the full low-resolution image and only masks the high-resolution context.

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
            [tokens] * batch_size + [null_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [null_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )


def make_model_fn(model, guidance_scale):
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    return model_fn


def sample_text_to_image(model, diffusion, prompt, batch_size, text_ctx,
                         image_size, guidance_scale, device):
    full_batch_size = batch_size * 2
    model_kwargs = build_text_kwargs(
        model, prompt, batch_size, text_ctx, device, include_null=True
    )
    model_fn = make_model_fn(model, guidance_scale)

    model.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, image_size, image_size),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model.del_cache()
    return samples


def sample_clip_guided(model, diffusion, clip_model, prompt, batch_size,
                       text_ctx, image_size, clip_guidance_scale, device):
    with th.no_grad():
        z_t = clip_model.text_embeddings([prompt] * batch_size)

    def cond_fn(x, t, grad_scale=clip_guidance_scale, **kwargs):
        with th.enable_grad():
            x_var = x.detach().requires_grad_(True)
            z_i = clip_model.image_embeddings(x_var, t)
            loss = th.exp(clip_model.logit_scale) * (z_t * z_i).sum()
            grad = th.autograd.grad(loss, x_var)[0].detach()
        return grad * grad_scale

    model_kwargs = build_text_kwargs(
        model, prompt, batch_size, text_ctx, device, include_null=False
    )
    return diffusion.p_sample_loop(
        model,
        (batch_size, 3, image_size, image_size),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
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


def upsample_image(model_up, diffusion_up, prompt, low_res, batch_size,
                   text_ctx, image_size, upsample_temp, device):
    upsample_kwargs = build_upsample_kwargs(
        model_up, prompt, low_res, batch_size, text_ctx, device
    )
    up_shape = (batch_size, 3, image_size, image_size)

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
    return up_samples


def sample_inpaint(model, diffusion, prompt, source_image, source_mask,
                   batch_size, text_ctx, image_size, guidance_scale, device):
    full_batch_size = batch_size * 2
    model_kwargs = build_text_kwargs(
        model, prompt, batch_size, text_ctx, device, include_null=True
    )
    model_kwargs.update(
        inpaint_image=(source_image * source_mask)
        .repeat(full_batch_size, 1, 1, 1)
        .to(device),
        inpaint_mask=source_mask.repeat(full_batch_size, 1, 1, 1).to(device),
    )

    def denoised_fn(x_start):
        return (
            x_start * (1 - model_kwargs["inpaint_mask"])
            + model_kwargs["inpaint_image"] * model_kwargs["inpaint_mask"]
        )

    model_fn = make_model_fn(model, guidance_scale)
    model.del_cache()
    edited = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, image_size, image_size),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]
    model.del_cache()
    return edited
```
