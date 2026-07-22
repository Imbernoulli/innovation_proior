I want a model that takes a free-form caption and produces a photorealistic image, and I want the same machinery to edit an existing image by filling a masked region from text. The strongest image generator I have in hand is no longer a GAN for the setting I care about; guided diffusion is the thing producing the best class-conditional samples. So one path is tempting: keep the diffusion backbone, replace the class label with language, and find a guidance signal that can push samples toward a sentence. Whether that path actually works hinges on details I have not checked yet — the exact form of the guidance term, the sign of the score relation, the way text enters the network — so let me work through them rather than assume they fall into place.

Let me pin down the diffusion object before I touch text. The forward chain is `q(x_t|x_{t-1}) = N(x_t; √α_t x_{t-1}, (1−α_t)I)`, and by composing the steps I can write `x_t = √\bar α_t x_0 + σ_t ε`, with `σ_t = √(1-\bar α_t)` and `ε ∼ N(0,I)`. If the final accumulated noise is large enough, `x_T` is almost standard Gaussian, and if each step is small enough, the reverse posterior is close to Gaussian. I train `p_θ(x_{t-1}|x_t) = N(μ_θ(x_t), Σ_θ(x_t))` and sample by starting from noise and walking backward.

The practical loss is not the raw variational bound. I sample a timestep, add known Gaussian noise, and regress that noise:

`L_simple = E_{t,x_0,ε}[‖ε - ε_θ(x_t,t)‖²]`.

The sign matters here, and it is exactly the kind of thing I tend to get backwards, so let me actually differentiate rather than quote a result. The corruption is Gaussian, `q(x_t|x_0) = N(x_t; √\bar α_t x_0, σ_t² I)`, so `log q = -(x_t - √\bar α_t x_0)²/(2σ_t²) + const` and
`∇_{x_t} log q(x_t|x_0) = -(x_t - √\bar α_t x_0)/σ_t²`.
Now substitute `x_t = √\bar α_t x_0 + σ_t ε`, which makes the numerator `x_t - √\bar α_t x_0 = σ_t ε`, so the whole thing collapses to `-σ_t ε / σ_t² = -ε/σ_t`. So an `ε`-predictor is a score model with a known *negative* scale: `∇_{x_t} log p_t(x_t) ≈ -ε_θ(x_t,t)/σ_t`. The minus sign is real, not cosmetic — if I drop it the guidance derivation looks cleaner but every guided step pushes in the wrong direction. I'll carry it through carefully. I can still use the `ε` parameterization because the reverse mean `μ_θ` is recovered from it, and I should learn the variance too; learned `Σ_θ` is what lets the sampler take far fewer reverse steps without falling apart.

Guidance is the next pressure point. Sampling from the conditional denoiser alone gives diversity, but I need a knob that increases fidelity and caption match at the cost of diversity. Classifier guidance gives exactly such a knob, and I want to see precisely where the `Σ` factor in its mean perturbation comes from rather than copy the formula. The reverse step is locally `p(x_{t-1}|x_t) ≈ N(μ, Σ)`, and guidance multiplies it by the classifier raised to a power, `p_φ(y|x_{t-1})^s`. Near `μ`, expand the log-classifier to first order: `s log p_φ(y|x_{t-1}) ≈ const + s(x_{t-1}-μ)^T g` with `g = ∇ log p_φ(y|x)|_μ`. Adding the two logs,
`-½(x-μ)^T Σ^{-1}(x-μ) + s(x-μ)^T g`,
and expanding the quadratic gives `-½ x^T Σ^{-1} x + x^T(Σ^{-1}μ + s g) + const`. That is again a Gaussian; matching it to `-½ x^T Σ^{-1} x + x^T Σ^{-1} μ'` reads off `Σ^{-1} μ' = Σ^{-1}μ + s g`, i.e.
`μ' = μ + s Σ g`.
Let me sanity-check that against the stationarity condition directly: the gradient of the combined log-density is `-Σ^{-1}(x-μ) + s g`, which vanishes at `x = μ + sΣg` — same answer. So the `Σ` factor is not decoration; it is exactly the matrix that turns the gradient `g` into a displacement of the mean:

`μ̂_θ(x_t|y) = μ_θ(x_t|y) + s · Σ_θ(x_t|y) ∇_{x_t} log p_φ(y|x_t)`.

For classes, this is clean. For captions, it is awkward. A caption is not one label in a closed set, and training a classifier over arbitrary sentences at every noise level is the wrong object. I need the classifier-gradient effect without a separate classifier.

Bayes gives me one. If the generative model knows both `p(x_t|c)` and `p(x_t)`, then an implicit classifier is `p^i(c|x_t) ∝ p(x_t|c)/p(x_t)`. Its gradient is

`∇_{x_t} log p^i(c|x_t) = ∇ log p(x_t|c) - ∇ log p(x_t)`.

Using the score/noise relation, this becomes

`∇ log p^i(c|x_t) ≈ -(ε*(x_t|c) - ε*(x_t|∅))/σ_t`.

That minus sign in `score ≈ -ε/σ_t` is what worries me: I want a guidance rule expressed in the network's `ε` output, but the clean extrapolation lives in score space. A guided score that starts from the unconditional score and extrapolates toward the conditional one is

`score_guided = score_∅ + w(score_c - score_∅)`, with `w ≥ 1`.

I need to push this through the `-1/σ_t` map and see what survives. Substitute `score_c = -ε_c/σ_t` and `score_∅ = -ε_∅/σ_t`:

`score_guided = -ε_∅/σ_t + w(-ε_c/σ_t + ε_∅/σ_t) = -(1/σ_t)[ε_∅ + w(ε_c - ε_∅)]`.

To go back to the noise parameterization I invert the same relation, `ε = -σ_t · score`, so

`ε̂_θ(x_t|c) = -σ_t · score_guided = ε_∅ + w(ε_c - ε_∅)`.

I half-expected the two minus signs to leave a sign error somewhere, but they cancel exactly: the `-σ_t` from the inversion and the `-1/σ_t` inside the score relation multiply to `+1`, so the `ε`-space rule comes out with a plain plus sign,
`ε̂_θ(x_t|c) = ε_θ(x_t|∅) + w · (ε_θ(x_t|c) - ε_θ(x_t|∅))`,
even though the score relation it descends from carries a minus. That is reassuring — it means I can implement guidance entirely in the `ε` output with no per-step `σ_t` bookkeeping. I only need the network to produce both terms. The simplest way is to make the empty caption a real training condition: during a fine-tune, replace the token sequence with the empty sequence `20%` of the time. If the drop rate is too small, the unconditional branch is weak; if it is too large, I waste conditional training. A modest drop gives the same weights both behaviors, and at sampling time I run the denoiser once on a doubled batch, half captioned and half empty.

CLIP is the other plausible route. It already gives an image embedding and a text embedding whose dot product is a caption-match score, so I can use `∇_{x_t}(f(x_t)·g(c))` as the mean-perturbing gradient. But the classifier-guidance lesson repeats: the guidance model is evaluated on noisy intermediate images, so a clean-image CLIP is out of distribution. If I use CLIP, I should train a noised image encoder `f(x_t,t)` with the same contrastive objective and the same noise schedule as the diffusion model. Then the CLIP route is a real analogue of classifier guidance:

`μ̂_θ(x_t|c) = μ_θ(x_t|c) + s · Σ_θ(x_t|c) ∇_{x_t}(f(x_t,t)·g(c))`.

I now have two text guidance mechanisms to compare: the external noised-CLIP gradient and the model's own conditional-minus-unconditional direction. Structurally I expect the second to be stronger because it uses the denoiser's full learned distribution rather than a separate similarity model, and it avoids training another noise-aware network.

Text also has to enter the denoiser. A single pooled sentence vector is too lossy; it can carry global topic, but spatial image features need to bind words, attributes, and relations at many resolutions. The ADM backbone already has attention blocks where spatial features can look at context, so I encode the caption with a Transformer, use the final token embedding where the class embedding used to go, and also project the full sequence of token embeddings into each attention layer's width and concatenate those tokens to the attention context. The final token gives global conditioning everywhere; the full sequence gives each attention layer access to the words it needs.

Generating directly at high resolution would make every reverse step expensive. I keep the base model at `64×64`, then use a separate text-conditional super-resolution diffusion model for `64×64 → 256×256`, conditioning on the low-resolution image by channel concatenation. This keeps semantic generation and high-frequency refinement separated.

Editing needs one more adjustment. The training-free inpainting recipe overwrites the known region with a noised version of the original after each sampling step, but then the model only sees noisy context around the hole. That is exactly where seams and edge artifacts come from. I should fine-tune for the editing task itself: erase random regions, feed the clean known RGB pixels and a mask as extra channels, and train the model to denoise the completed image. Four new input channels are enough: three for the known image and one for the mask. I initialize those new input weights to zero so the fine-tune starts as the original generator and gradually learns to use the editing context. For the upsampler, I should keep the whole low-resolution image available because it carries the coarse scene, and mask only the high-resolution context.

In the sampler I still need a projection that keeps the known pixels exactly fixed while letting the model fill the hole, and the convention here is easy to invert by mistake, so let me pin it down on a concrete pair of pixels. Let the mask be `1` on known pixels and `0` on the hole, and store the masked-out original as `inpaint_image = source · mask`. Take pixel A known with original value `0.7` (`mask=1`) and pixel B in the hole (`mask=0`), and suppose the model's predicted clean image is `x_start = [0.1, 0.9]`. Then `inpaint_image = [0.7·1, (−0.4)·0] = [0.7, 0]`. The projection `x_start·(1−mask) + inpaint_image·mask` gives, pixel by pixel: A `= 0.1·(1−1) + 0.7·1 = 0.7`, and B `= 0.9·(1−0) + 0·0 = 0.9`. So A is forced back to its true `0.7` regardless of what the model predicted, and B keeps the model's `0.9`. That is the behavior I want, and it confirms the orientation: `mask=1` means "trust the original," `mask=0` means "trust the model." Had I swapped the convention, the projection would have frozen the hole and regenerated the context — the exact opposite — so checking it on these two pixels is worth the line.

At sampling time I make the doubled batch explicit: one half carries the caption, the other carries the empty sequence. I duplicate the noisy latents, split the first three output channels as `ε`, extrapolate the conditional and unconditional halves, leave the variance channels alone, and let the diffusion sampler convert that guided `ε` into the reverse mean.

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
```

If I choose the CLIP route instead, the sampler's existing `cond_fn` hook carries the mean perturbation. The noised CLIP model normalizes image and text embeddings, multiplies their dot product by the learned logit scale, differentiates with respect to the current noisy image, and returns the scaled gradient.

```python
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
```

For the high-resolution stage, I keep the generated `64×64` sample as an explicit low-resolution condition and pass it to the text-conditioned upsampler. The strided fast upsampling schedule fits the DDIM loop.

```python
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
```

For editing, the same classifier-free base sampler gets extra conditioning channels and a projection on the predicted clean image. The mask convention is explicit in the algebra: known pixels are copied from `inpaint_image`, unknown pixels remain the sampler's prediction.

```python
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
