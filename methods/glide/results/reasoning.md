Let me start from what I want and what's actually working right now. I want a model that takes a free-form caption and produces a photorealistic image, and ideally the same model should be able to *edit* an existing image from text — fill in a masked region with something the caption describes. The thing that's working best for image synthesis at the moment is diffusion: class-conditional diffusion models have overtaken GANs on the standard fidelity benchmarks. So the natural play is to take a diffusion model and condition it on text instead of a class label. But two questions immediately stand in the way: how do I inject something as rich as a sentence into the denoiser, and — the one that actually decides sample quality — how do I *guide* the reverse process toward the caption?

Let me get the diffusion machinery straight first, because everything hangs off it. The forward process noises a clean image `x_0` step by step, `q(x_t|x_{t-1}) = N(x_t; √α_t x_{t-1}, (1−α_t)I)`. If each step adds only a little noise, the true reverse `q(x_{t-1}|x_t)` is close to a diagonal Gaussian, and if the chain adds enough total noise, `x_T` is essentially `N(0,I)`. So I learn a reverse model `p_θ(x_{t-1}|x_t) = N(μ_θ(x_t), Σ_θ(x_t))` and generate by starting from pure noise and denoising down. I won't optimize the variational bound directly; the surrogate that works is to noise `x_0` to `x_t` with a known `ε ∼ N(0,I)` and have the network predict that noise, `L_simple = E_{t,x_0,ε}[ ‖ε − ε_θ(x_t,t)‖² ]`. From `ε_θ` I can recover `μ_θ`. And there's a fact I'll lean on hard later: the noise prediction is, up to scale, the *score* of the noised data, `∇_{x_t} log p(x_t) ∝ ε_θ(x_t,t)`. I'll also let the model learn `Σ_θ` rather than fix it, because a learned variance lets me sample well with far fewer steps — which matters a lot when each step is a forward pass of a multi-billion-parameter net. So the denoiser's output is really two things per pixel: a predicted noise and a variance.

Now, guidance. Why is guidance even needed — why not just sample from the conditional model `ε_θ(x_t|y)` and be done? Because, empirically, unguided conditional diffusion samples are more diverse but less sharp, and there's a knob that trades diversity for fidelity. The known way to turn that knob is **classifier guidance**: take a classifier `p_φ(y|x_t)` trained on noised images and nudge the reverse mean up the gradient of its log-probability,

`μ̂_θ(x_t|y) = μ_θ(x_t|y) + s · Σ_θ(x_t|y) ∇_{x_t} log p_φ(y|x_t)`,

with a scale `s`: bigger `s` means the sample is pushed harder toward "things the classifier is confident are class `y`," which sharpens fidelity at the cost of diversity. This is clean for a fixed label set. But my conditioning signal is *text*, an open-ended caption. Training a classifier `p_φ(caption | x_t)` over arbitrary sentences is awkward — a sentence isn't a class, and a classifier for text-given-image is exactly the kind of model I'd rather not have to build and train on noised images. So classifier guidance doesn't transplant cleanly to text. I need guidance that doesn't lean on a separate classifier.

Here's the reframe. What classifier guidance actually does is add `∇_{x_t} log p(y|x_t)` to the score. Can I get that gradient *without* a classifier? Consider an implicit classifier built straight out of the generative model itself by Bayes' rule: `p^i(y|x_t) ∝ p(x_t|y)/p(x_t)`. Take its gradient w.r.t. `x_t`:

`∇_{x_t} log p^i(y|x_t) ∝ ∇_{x_t} log p(x_t|y) − ∇_{x_t} log p(x_t)`.

And I just said the score is the noise prediction — so `∇_{x_t} log p(x_t|y) ∝ ε*(x_t|y)` (conditional score) and `∇_{x_t} log p(x_t) ∝ ε*(x_t)` (unconditional score). The implicit-classifier gradient is therefore proportional to the *difference of the model's own conditional and unconditional noise predictions*, `ε*(x_t|y) − ε*(x_t)`. So I don't need any external classifier at all: if my one network can produce both a conditional `ε_θ(x_t|y)` and an unconditional `ε_θ(x_t|∅)`, then adding `s` times their difference to the prediction *is* classifier guidance with the model's own implicit classifier. Concretely the guided prediction is

`ε̂_θ(x_t|y) = ε_θ(x_t|∅) + s · ( ε_θ(x_t|y) − ε_θ(x_t|∅) )`, `s ≥ 1`,

which says: start from the unconditional prediction and extrapolate *past* the conditional one, away from "what you'd predict with no caption" and toward "what you'd predict given the caption." This is classifier-free guidance, and the same `s` knob trades diversity for fidelity, for the same reason as before — it's amplifying the implicit-classifier gradient.

For this to work the network has to know how to be unconditional. The trick is to train it to be both: with some fixed probability, replace the label by a null token `∅`. Then `ε_θ(·|∅)` is a genuine unconditional denoiser and `ε_θ(·|y)` the conditional one, sharing all weights. For text, `∅` is just the empty caption: sometimes during training I feed the model an empty token sequence instead of the real caption, so it learns to denoise with and without text. At sampling I run the network twice — once with the caption `c`, once with `∅` — and combine: `ε̂_θ(x_t|c) = ε_θ(x_t|∅) + s·(ε_θ(x_t|c) − ε_θ(x_t|∅))`. Two things make this attractive over classifier guidance. The model guides itself with its *own* learned knowledge rather than a separate, possibly weaker classifier; and conditioning on something hard to classify, like a sentence, becomes trivial — there's no classifier to build.

There's a competing option I should take seriously, because there's a natural text-image scorer lying around: CLIP. CLIP gives me an image encoder `f(x)` and a caption encoder `g(c)` whose dot product `f(x)·g(c)` is high when image and caption match. That dot product is exactly a "how well does this image match the caption" score — so I can drop it into classifier guidance in place of the classifier's log-prob, perturbing the mean by the gradient of the similarity:

`μ̂_θ(x_t|c) = μ_θ(x_t|c) + s · Σ_θ(x_t|c) ∇_{x_t} ( f(x_t) · g(c) )`.

But there's a subtlety that mirrors the classifier-guidance requirement. In classifier guidance, the classifier had to be trained on *noised* images, because during sampling it's evaluated on the noisy intermediate `x_t`, not on clean images. The same must hold for CLIP: the public CLIP was trained on clean images, so the blurry, noised `x_t` that show up all through the reverse process are out-of-distribution for it, and its gradients become unreliable — which is why people who guide with public CLIP have to bolt on data augmentations and perceptual losses to get anything recognizable. The fix is to train a **noised CLIP**: an image encoder `f(x_t, t)` that takes the noised image and timestep, trained with the same contrastive objective but on noised images with the same noise schedule as the base diffusion model. Then its similarity gradient is the right signal at every step, no augmentation tricks needed.

So I now have two viable guidance routes for text — classifier-free guidance (extrapolate the model's own conditional vs. unconditional `ε`) and CLIP guidance (gradient of a noised-CLIP similarity). Both turn the same fidelity/diversity knob. Which one to use is the open question; the thing I'd want to validate is which gives more photorealistic, more caption-faithful samples under human comparison. (My expectation, from the structure: classifier-free guidance uses the model's own full knowledge rather than the narrower view of a similarity score, and avoids the extra noised-CLIP encoder.)

Now the architecture for getting text in. The denoiser is the ADM UNet (the class-conditional diffusion backbone). I encode the caption into a sequence of `K` tokens with a Transformer, and use its outputs in two complementary places. The final token embedding stands in for the class embedding that ADM normally adds — a single global vector summarizing the caption, injected everywhere the class embedding was. And the full sequence of `K` token feature vectors gets separately projected to the width of each attention layer and concatenated onto that layer's attention context, so every attention block in the UNet can attend over the caption tokens, not just a pooled summary. Global conditioning plus per-layer cross-attention to the token sequence. The base model runs at `64×64`; a second text-conditional diffusion model upsamples `64×64 → 256×256` (conditioning on the low-res image by channel-concatenation, the SR3 recipe).

For the unconditional capability, after the main training run I fine-tune the base model exactly as in pretraining except that `20%` of the time the token sequence is the empty sequence. That teaches `ε_θ(·|∅)` without erasing the conditional behavior — the model keeps generating text-conditional outputs and also gains a clean unconditional mode, which is precisely what classifier-free guidance consumes.

Editing falls out with one more fine-tune. The training-free way to inpaint is to sample normally but overwrite the known region with a fresh `q(x_t|x_0)` noised sample of the original after each step. The problem: the model only ever sees a *noised* version of the surrounding context, so it can't condition cleanly on the real pixels around the hole, and you get edge artifacts. Better to teach it the task: during fine-tuning, erase random regions and feed the model the remaining (clean) pixels plus a mask channel as extra conditioning. Concretely I add four input channels to the UNet — a second set of RGB channels (the clean known region) and a mask channel — and initialize the new channels' input weights to zero so the model starts out behaving exactly as before and learns the inpainting use of them gradually. (For the upsampler I always give the full low-res image but only the unmasked part of the high-res.)

Now the sampling code. The clean way to do classifier-free guidance is to run the conditional and unconditional predictions in one batched forward pass: stack the same noisy latents twice, with the caption tokens on one half and the empty tokens on the other, run the UNet once, split off the `ε` channels (the first 3 of the 6), recombine by the guidance formula, and pass the guided `ε` (with the untouched variance channels) on to the sampler step.

```python
import torch as th

def model_fn(x_t, ts, **kwargs):
    # batch is [conditional half ; unconditional half]; run them together
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]          # 3 noise channels + variance
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    # ε̂ = ε(∅) + s·(ε(c) − ε(∅)): extrapolate away from unconditional, toward the caption
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)
```

`model_kwargs` carries the caption tokens (and attention mask) on the conditional half and the empty/uncond tokens on the unconditional half, so a single batched call gives me both predictions. Sampling is the standard reverse loop with no extra `cond_fn`, since the guidance already lives inside `model_fn`:

```python
samples = diffusion.p_sample_loop(
    model_fn,
    (full_batch_size, 3, image_size, image_size),
    clip_denoised=True,
    model_kwargs=model_kwargs,    # tokens for the cond half, empty tokens for the uncond half
    cond_fn=None,
)
```

The CLIP-guidance alternative instead supplies a `cond_fn` that perturbs the reverse mean. It differentiates the noised-CLIP similarity — image embedding of the current noisy `x_t` dotted with the text embedding `z_t` of the caption — with respect to `x_t`, and returns that gradient scaled by the guidance:

```python
def cond_fn(x, t, grad_scale=grad_scale, **kwargs):
    with th.enable_grad():
        x_var = x.detach().requires_grad_(True)
        z_i = self.image_embeddings(x_var, t)               # noised-CLIP image embed f(x_t, t)
        loss = th.exp(self.logit_scale) * (z_t * z_i).sum()  # f(x_t) · g(c)
        grad = th.autograd.grad(loss, x_var)[0].detach()
    return grad * grad_scale

samples = diffusion.p_sample_loop(
    model, (batch_size, 3, image_size, image_size),
    clip_denoised=True, model_kwargs=model_kwargs,
    cond_fn=cond_fn,                                        # μ̂ = μ + s·Σ·∇(f·g)
)
```

Tracing the chain back: I wanted text-to-image and text-editing, and diffusion was the strongest synthesis backbone, so I conditioned an ADM denoiser on a caption (global token embedding in place of the class embedding, plus per-attention-layer cross-attention over the token sequence) and trained it with `ε`-prediction and a learned variance. Guidance is what sets fidelity, and classifier guidance doesn't transplant to text because a sentence isn't a class — but the implicit classifier `p(x|y)/p(x)` shows that the classifier gradient equals the difference of the model's own conditional and unconditional scores, so by training the model to also denoise unconditionally (drop the caption `20%` of the time) I get classifier-free guidance for free: `ε̂ = ε(∅) + s·(ε(c) − ε(∅))`. The CLIP-guidance alternative replaces the classifier with a noised-CLIP similarity gradient (noised because the intermediate `x_t` are out-of-distribution for clean-image CLIP). A text-conditional upsampler takes `64×64` to `256×256`, and an inpainting fine-tune with extra RGB+mask channels (zero-initialized) turns the same model into a text-driven editor.
