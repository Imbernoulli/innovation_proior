# GLIDE — synthesis (arXiv 2112.10741, verified — Nichol, Dhariwal, Ramesh, Shyam, Mishkin, McGrew, Sutskever, Chen, 2021/2022)

## Pain point / goal
Text-conditional photorealistic image generation + editing. Want a single model that turns free-form text into realistic images AND edits existing images (inpainting). Prior text-to-image = GANs (AttnGAN etc) or autoregressive over discrete latent codes (DALL-E, building on VQ-VAE). Diffusion models had become SOTA for class-conditional image synthesis (ADM, Dhariwal & Nichol 2021 beat GANs) — but how to push them to text, and how to trade off fidelity vs diversity vs caption-match? Central question: which GUIDANCE strategy works best for text — CLIP guidance or classifier-free guidance?

## Background: diffusion models (Sohl-Dickstein 2015; DDPM Ho 2020; improved Nichol&Dhariwal 2021)
Forward: q(x_t|x_{t-1}) = N(x_t; √α_t x_{t-1}, (1-α_t) I). Small 1-α_t per step → q(x_{t-1}|x_t) ≈ diagonal Gaussian; total noise large → x_T ≈ N(0,I).
Reverse model: p_θ(x_{t-1}|x_t) = N(μ_θ(x_t), Σ_θ(x_t)). Sample x_T~N(0,I), denoise to x_0.
Training: sample x_t ~ q(x_t|x_0) by adding noise ε; predict ε with ε_θ:
   L_simple = E_{t,x_0,ε~N(0,I)} [ ||ε − ε_θ(x_t,t)||² ]  (re-weighted surrogate of the VLB).
DDPM: derive μ_θ from ε_θ, fix Σ_θ constant. Score connection: ∇_{x_t} log p(x_t) ∝ ε_θ(x_t,t). Improved (Nichol&Dhariwal): LEARN Σ_θ → high quality with fewer steps; adopted here.
Super-resolution diffusion (SR3): p_θ(y_{t-1}|y_t,x), condition on low-res x by concatenating bicubic upsample in channel dim.

## Guidance strategies (the three components)
### Classifier guidance (Dhariwal & Nichol 2021, ADM)
Class-conditional model μ_θ(x_t|y), Σ_θ(x_t|y); perturb the mean by gradient of a classifier's log-prob:
   μ̂_θ(x_t|y) = μ_θ(x_t|y) + s · Σ_θ(x_t|y) ∇_{x_t} log p_φ(y|x_t).
s = guidance scale; larger s → better quality, less diversity. Needs a separate classifier trained ON NOISED images.

### Classifier-free guidance (Ho & Salimans 2021)
No separate classifier. During training, replace label y with null ∅ with fixed probability. At sampling, extrapolate:
   ε̂_θ(x_t|y) = ε_θ(x_t|∅) + s · ( ε_θ(x_t|y) − ε_θ(x_t|∅) ),  s ≥ 1.
Derivation: implicit classifier p^i(y|x_t) ∝ p(x_t|y)/p(x_t). Its score:
   ∇_{x_t} log p^i(x_t|y) ∝ ∇log p(x_t|y) − ∇log p(x_t) ∝ ε*(x_t|y) − ε*(x_t).
So extrapolating ε away from the unconditional and toward the conditional = amplifying the implicit-classifier gradient (exactly classifier guidance but using the model's OWN conditional vs unconditional scores).
For text: replace caption c with empty sequence ∅ sometimes during training; guide:
   ε̂_θ(x_t|c) = ε_θ(x_t|∅) + s · ( ε_θ(x_t|c) − ε_θ(x_t|∅) ).
Two wins: (1) model uses its OWN knowledge, not a separate (smaller) classifier; (2) trivial to condition on things hard to classify (like text).

### CLIP guidance
CLIP (Radford 2021): image encoder f(x), caption encoder g(c); contrastive — high f(x)·g(c) for matched pairs. Replace the classifier in classifier guidance with CLIP:
   μ̂_θ(x_t|c) = μ_θ(x_t|c) + s · Σ_θ(x_t|c) ∇_{x_t} ( f(x_t) · g(c) ).
Must use a NOISED CLIP (image encoder f(x_t,t) trained on noised images), else the noised intermediate x_t are OOD for public CLIP → hurts quality (public CLIP needs augmentations/perceptual losses to work; noised CLIP needs no tricks).

## Training (architecture)
- ADM architecture (Dhariwal & Nichol) augmented with text. Base: 3.5B params (visual ~2.3B, width 512 channels; ImageNet 64×64 ADM scaled), text Transformer 24 residual blocks width 2048 (~1.2B). Predicts p(x_{t-1}|x_t, c) at 64×64.
- Text conditioning: encode caption → K tokens → Transformer. Use output two ways: (1) final token embedding replaces the class embedding in ADM; (2) last layer of K token embeddings projected to each attention layer's dim and concatenated to attention context at every layer.
- Upsampler: 1.5B params, 64×64 → 256×256, text-conditioned (smaller text encoder width 1024), ADM ImageNet upsampler with 384 base channels.
- Train base 2.5M iters batch 2048; upsampler 1.6M iters batch 512; 16-bit + loss scaling. Dataset = same as DALL-E.
- Fine-tune base for classifier-free guidance: same as pretraining but 20% of token sequences → empty sequence (retains conditional ability + can generate unconditionally).
- Learned Σ_θ (improved diffusion) → model output = 6 channels for RGB: 3 for ε, 3 for variance interpolation.

## Inpainting / editing
Naive (training-free): sample as usual but replace known region with q(x_t|x_0) sample after each step — model sees only noised context → edge artifacts. Better: fine-tune for inpainting (like Palette): erase random regions, feed remaining + mask channel. Add 4 input channels: a second RGB set + a mask channel; init new input weights to zero. Upsampler: always full low-res, only unmasked high-res. Enables text-driven editing, SDEdit sketch→image, iterative scene building.

## Noised CLIP
Train CLIP image encoder f(x_t,t) on noised images, same objective as CLIP, 64×64, same noise schedule as base. ViT-L, patch 4×4. Without noise-awareness, intermediate samples OOD → poor quality.

## Sampling hyperparameters
- Base: 150 diffusion steps (paper samples), 100 for inpainting, 250 for evals (slight FID boost).
- Upsampler: strided schedule, 27 steps total: split into 5 segments, sample 10,10,3,2,2 evenly-spaced steps per segment (more steps at low-noise end (0,200], fewer at (800,1000]). Found by FID sweep.
- Guidance scales (samples): CLIP guidance scale 2.0, classifier-free guidance scale 3.0. Small 300M model prefers CF scale 4.0.
- Finding: classifier-free guidance preferred by humans over CLIP guidance for BOTH photorealism and caption-match; CF gives bigger Elo boost than 10× model size.

## Code grounding (openai/glide-text2im)
Classifier-free guidance (notebooks/text2im) — model_fn:
```
def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)        # batch = [cond; uncond]
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]  # 3 eps channels + variance
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)
```
model_kwargs: tokens (caption) for the cond half, empty/uncond_tokens for the uncond half (double batch); mask. Then diffusion.p_sample_loop(model_fn, (full_batch,3,H,W), clip_denoised=True, model_kwargs=..., cond_fn=None).
CLIP guidance (notebooks/clip_guided, clip/model_creation.py cond_fn):
```
def cond_fn(x, t, grad_scale=grad_scale, **kwargs):
    with torch.enable_grad():
        x_var = x.detach().requires_grad_(True)
        z_i = self.image_embeddings(x_var, t)         # noised CLIP image embed f(x_t,t)
        loss = torch.exp(self.logit_scale) * (z_t * z_i).sum()  # f(x_t)·g(c) (z_t = text embed)
        grad = torch.autograd.grad(loss, x_var)[0].detach()
    return grad * grad_scale
```
cond_fn = clip_model.cond_fn([prompt]*batch, guidance_scale); passed to p_sample_loop as cond_fn (perturbs mean by Σ·grad inside the sampler).

## Design decisions → why
- Diffusion over GAN/AR: SOTA fidelity for class-conditional; exact-ish likelihood training; one model for gen + editing.
- Predict ε with L_simple: DDPM surrogate, stable, score-equivalent.
- Learned Σ_θ: fewer sampling steps at high quality (improved diffusion).
- Text → Transformer → (class-emb replacement + per-attention-layer concat): inject caption both globally and at every attention layer.
- Classifier-free over classifier/CLIP guidance: uses model's own knowledge, no separate classifier, handles text (hard to classify); empirically wins on photorealism + caption-match. Form derived from implicit classifier p(x|y)/p(x).
- Guidance scale s≥1, large s → fidelity up, diversity down (same tradeoff as classifier guidance).
- 20% caption dropout in fine-tune: teach the unconditional ε_θ(·|∅) without losing conditional ability.
- Noised CLIP: intermediate x_t are noisy → must be in-distribution for the guiding model; public CLIP OOD → needs hacks, noised CLIP doesn't.
- Inpainting fine-tune + mask channel + zero-init new weights: model sees full clean context (not just noised) → no edge artifacts; zero-init preserves pretrained behavior at start.
- Upsampler strided 10,10,3,2,2: most refinement needed at low-noise end; few steps suffice at high-noise end → 27 steps total.
