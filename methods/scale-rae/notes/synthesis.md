# Synthesis — Scaling RAE to Text-to-Image (working name: scale-rae)

## What is the method, in one breath
Build a large-scale text-to-image generator that runs flow-matching diffusion **directly in the
frozen semantic representation space** of a pretrained image encoder (SigLIP-2 So400M, patch-14,
producing N=256 tokens of dimension d=1152 at 224x224), not in a compressed VAE space. A pretrained
LLM (Qwen2.5) is conditioned on the text prompt plus a block of learnable query tokens (MetaQuery
style); the query-token outputs, projected by a small MLP, condition a DiT (LightningDiT-style) that
generates the 256x1152 representation; a separately trained ViT decoder (the Representation
Autoencoder, RAE, decoder) renders those features to pixels. The same frozen encoder also feeds the
understanding path, so understanding and generation live in one shared latent space.

## Empirical-first character
This is an empirical-first investigation. The central question is "can the RAE recipe — proven only
on class-conditional ImageNet — survive the move to open-ended, large-scale T2I?" The contribution is
the *answer plus the simplified recipe*: a set of diagnostic findings about which RAE design choices
are fundamental vs. crutches for small models, and the resulting clean recipe. So the load-bearing
"findings" are mostly diagnostic/motivating (about existing RAE design choices and about VAE vs.
representation latents), and they belong in context/background. The *proposed recipe's own benchmark
wins* (RAE beats VAE at every scale, 4x convergence, finetune overfitting curves) are the proposed
method's evaluation outcomes and stay out of context/reasoning.

Diagnostic / motivating findings (pre-method facts about existing systems — CONTEXT-eligible):
- High-dim representation latents were historically deemed "too abstract" or "intractable" for
  generative modeling (ImprovedDiffusion, prior work).
- Parent RAE work's own diagnostic facts (prior art here): (a) standard DiT FAILS to converge when
  its width d < latent token dimension n — provable lower bound L >= sum_{i=d+1}^n lambda_i (tail
  eigenvalues of cov(eps - x)); (b) Gaussian-noise diffusion "fills" the support to full rank, so
  capacity must scale with full dimension; (c) without dimension-aware noise schedule shift, gFID
  collapses (23.08 vs 4.81 in parent ImageNet study).
- Decoder-data diagnostics (this investigation, pre-method facts about decoders): scaling decoder
  data beyond ImageNet barely helps ImageNet rFID, moderately helps diverse web (YFCC), but **text
  glyph reconstruction needs text-specific data** — composition > scale. (Table: ImageNet-only Text
  rFID 2.64 -> +text data 1.62.)
- Saturation diagnostics: the wide DDT head gives a large boost (+11.2 GenEval) only at 0.5B where
  backbone width barely exceeds latent dim; vanishes by 2.4B+. Noise-augmented decoding helps early
  in training (<~15k steps) then becomes negligible.
- VAE-vs-representation latent diagnostics for unified models: VAE latents are insufficient for
  perception (so two-tower designs keep a separate CLIP encoder); representation latents serve both.

PROPOSED-method evaluation outcomes (EXCLUDE from context & reasoning conclusions):
- RAE beats FLUX-VAE at all DiT scales 0.5B-9.8B; 4.0x GenEval / 4.6x DPG convergence speedup;
  VAE overfits after 64 epochs while RAE stable to 256/512 epochs; latent TTS best-of-N gains;
  understanding unaffected by adding generation.

## Load-bearing ancestors (with the gap each leaves)

### Latent Diffusion (Rombach et al. 2022, "LDM") + VAE (Kingma & Welling 2014)
Idea: don't diffuse in pixel space; train a VAE to compress images into a small latent (channels
< 64), run the diffusion U-Net/transformer there, decode back. Gap it leaves: aggressive compression
discards information (reconstruction ceiling) and the latent is *not semantic* — it is tuned only for
pixel reconstruction, so a unified model that wants to both see and generate needs a *second*
(semantic) encoder. The compressed latent is also low-dim, which (counterintuitively) is exactly what
keeps standard DiTs trainable.

### DiT (Peebles & Xie 2023)
Idea: replace the diffusion U-Net with a plain transformer over latent patch tokens; condition on
timestep + class via adaptive LayerNorm (adaLN-zero) — predict the modulation (shift, scale, gate)
from the conditioning vector and apply it around attention and MLP. Scales cleanly with width/depth.
Gap: designed for low-dim VAE latents; nobody had shown it works when the *token dimension itself* is
huge (1000+).

### Flow matching / Rectified Flow (Lipman et al. 2023; Liu et al. 2023; Esser et al. 2024 "SD3")
Idea: define a straight-line probability path x_t = (1-t) x + t eps between data x (t=0) and noise eps
(t=1); train a network to predict the constant velocity v = eps - x (= dx_t/dt) via simple MSE; sample
by integrating the ODE backward from noise with Euler steps. Simpler & faster-converging than the
DDPM noise-prediction parameterization. SD3 contribution we reuse: **resolution-dependent timestep
shift** — for higher-resolution (more-pixel) images the per-pixel SNR at a given t is higher, so you
must push the schedule toward more noise. They derive sigma(t,n) = (t/(1-t)) * sqrt(1/n) for recovering
a constant signal from n pixels, and the reparam t_m = alpha t_n / (1 + (alpha-1) t_n) with
alpha = sqrt(m/n). Gap: SD3 derived this for *spatial resolution* in a low-channel VAE; it does not
address high *channel* dimension.

### Representation Autoencoder (RAE) — Zheng, Ma, Tong, Xie 2025 (the direct parent)
Idea: freeze a pretrained representation encoder (DINOv2 / SigLIP / MAE), train only a ViT decoder to
map its high-dim tokens back to pixels (losses: L1 + LPIPS + adversarial, encoder frozen). Then run
DiT diffusion directly on those semantic tokens. Three enabling design choices it introduced for
ImageNet:
1. **Width >= dimension.** Proves (Thm 1) that a DiT of width d < latent token dim n has training loss
   bounded below by sum_{i=d+1}^n lambda_i (tail eigenvalues of cov(eps - x)); intuition: noise fills
   the manifold to full rank so the model must carry the full dimension. Hence DiT hidden size must
   exceed 1152.
2. **Wide DDT head (DiT^DH).** Because ImageNet DiTs are often narrower (~1024) than the latent dim,
   widening the whole backbone is quadratically expensive; instead append a shallow but wide head
   (width 2688): backbone M(x_t|t,y)->conditioning z_t, head H(x_t|z_t,t)->velocity. A cheap way to buy
   the required width.
3. **Dimension-dependent noise schedule shift.** Generalize SD3's shift from spatial resolution to
   *effective data dimension* m = N*d (tokens x channels). With base n=4096, alpha=sqrt(m/n) and the
   same reparam t_m = alpha t_n/(1+(alpha-1) t_n). Without it ImageNet gFID 23.08 -> 4.81 with it.
4. **Noise-augmented decoding.** Train the decoder not only on clean encoder tokens z but on z+noise,
   n~N(0, sigma^2 I), sigma ~ |N(0, tau^2)|, to bridge the gap between the decoder's training inputs
   (true encoder tokens) and inference inputs (slightly-off diffusion samples).
Gap it leaves: only validated on ImageNet — fixed resolution, curated, class-conditional. Unknown
whether (a) the decoder generalizes to open-world images / text glyphs, (b) which of the 4 choices are
fundamental vs. small-scale crutches, (c) whether the convergence/quality advantage over VAE survives
billion-parameter T2I.

### MetaQuery (Pan et al. 2025)
Idea: a clean interface between a (possibly frozen) multimodal LLM and a diffusion model — prepend a
fixed block of learnable query tokens to the text; the LLM processes text+queries; the query-token
hidden states, projected by an MLP connector, become the diffusion model's conditioning. Trained with
ordinary image-caption pairs + standard diffusion loss. Earlier MetaQuery reported limited benefit
from scaling/finetuning the LLM. Gap: used with VAE-latent diffusion; never paired with a
representation-space diffusion target, and the LLM-scaling pessimism is worth re-testing at large DiT
scale + finetuned LLM.

### LightningDiT (Yao et al. 2025)
Idea: a modernized DiT block (RMSNorm, SwiGLU FFN, RoPE on the token grid, QK-norm optional,
adaLN conditioning, optionally no-shift adaLN) plus better training recipe; big convergence speedups.
We adopt this block as the DiT backbone. Gap: still a recipe, not a method on its own.

### SigLIP-2 (Tschannen et al. 2025) — the frozen encoder
Sigmoid-loss vision-language encoder with added self-supervised + decoder losses giving strong dense
patch features. So400M patch-14 at 224 -> 16x16=256 tokens, d=1152. Why it works as the RAE encoder:
language supervision + dense-feature objectives yield a semantically structured, high-dim space that
is good both for understanding and (once a decoder is trained) for reconstruction.

## Design-decision -> why table (the heart of reasoning depth)

| Decision | Why this, not the obvious alternative |
|---|---|
| Diffuse in frozen representation space, not VAE latent | VAE latent is compressed + non-semantic; need a 2nd encoder for understanding. Representation space is semantic AND high-fidelity once a decoder is trained -> one shared space for see+generate. Also empirically faster-converging in the parent ImageNet study. |
| Freeze encoder, train only decoder (RAE) | Finetuning the encoder would destroy the semantic structure that makes understanding work and makes the space shared. A lightweight decoder suffices to invert frozen tokens to pixels. |
| SigLIP-2 So400M p14 -> 256 tokens, d=1152 | 256 tokens matches a manageable query budget; language supervision gives a space aligned to text (good for T2I conditioning) and dense-feature training gives invertible patch tokens. (Ablate: WebSSL-DINO works too -> robustness to encoder.) |
| Flow matching (rectified flow), velocity pred, linear path | Straight-line path + constant velocity target = simplest, fastest-converging diffusion parameterization; matches modern SOTA (SD3/FLUX) and the parent RAE. |
| Dimension-dependent noise shift, base n=4096, alpha=sqrt(m/n) | High channel dim raises per-token SNR at a given t (same SNR argument as SD3 resolution shift, generalized from pixels to m=N*d). Without the shift the model sees too little effective noise and fails (diagnostic: 23.6 vs 49.6 GenEval). This is the ONE RAE choice that stays essential at scale. |
| Standard DiT, NO wide DDT head | DDT existed only to buy width when backbone (~1024) < latent dim (1152). Large T2I DiTs are already wide (>=2048 >> 1152), so the bottleneck is gone; DDT's +11.2 boost at 0.5B vanishes by 2.4B. Drop it -> simpler. |
| Keep DiT hidden size > latent dim even at 0.5B | The parent width>=dim theorem: width below latent dim has an irreducible loss floor = tail eigenvalues. So DiT-0.5B uses hidden 1280 > 1152. |
| NO noise-augmented decoding at scale | It is regularization that matters only before the diffusion model has learned a robust latent manifold (<~15k steps); at T2I scale the model converges past that quickly and the gain saturates. Drop it -> simpler. (tau capped at 0.2 anyway since high tau stalls decoder training.) |
| MetaQuery interface: learnable queries + MLP connector, finetune the LLM | Need to turn a text-LLM into a generation conditioner. Learnable queries read the prompt and become per-token conditioning; #queries=256 to match the 256 visual tokens. Finetune (not freeze) the LLM + use a large DiT -> recovers LLM-scaling gains that frozen MetaQuery missed. |
| Per-token adaLN conditioning (c = t_embed + query_embed, then SiLU) | The query outputs are a *sequence* (256 tokens), aligned 1:1 with the 256 latent tokens, so conditioning is per-token (y has shape [B,L,D]) rather than a single pooled vector -> richer spatial/semantic control. |
| Separate optimizers / LRs for LLM vs DiT | LLM is pretrained (small LR 5e-5) and DiT is from scratch (large LR 5e-4); coupling them destabilizes training. Decouple betas too (LLM 0.9/0.999, DiT 0.9/0.95). |
| Width-over-depth when scaling DiT | Recent vision-scaling findings favor width; also keeps hidden > latent dim guaranteed. |
| Decoder losses L1 + LPIPS + Gram + adversarial; DINO-S/16 discriminator | L1=pixel fidelity, LPIPS=perceptual, Gram=texture (helps reconstruction per ATOKEN), adversarial=sharpness. DINO-S/8 (parent) won't converge on web images; S/16 is a stronger/stabler discriminator at web scale. omega_G=100, omega_L=1, omega_A=10. |
| Two-stage train: pretrain from scratch + finetune on small HQ set | Standard T2I recipe (Emu/SDXL/PixArt). From scratch (not init from a released checkpoint) to make RAE-vs-VAE convergence comparison fair. |
| Latent-space test-time scaling (shared space payoff) | Because generation outputs land in the *same* space the LLM understands, the LLM can score its own generated latents directly (prompt-confidence or yes-logit), best-of-N, without decode->re-encode. Only possible because of the shared representation space. |

## Canonical code anchors (from official repo ZitengWangNYU/Scale-RAE)
- `scale_rae/model/diffusion_loss/diffusion/rf.py`: RectifiedFlow. x_start=clean data, x_end=noise,
  alpha_t=1-t, sigma_t=t, x_t = (1-t)x_start + t x_end; target velocity = x_end - x_start; loss =
  mean_flat((pred-target)^2). get_timestep: logit-normal u~N(mean,std), t=sigmoid(u), then shift
  t = alpha t/(1+(alpha-1) t) with alpha = 1/size_ratio (= sqrt(m/n)). euler_sample: x_s = x_t +
  (sigma_s - sigma_t) * model_pred, integrating from t=1 (noise) to 0 (data) so delta<0.
- `scale_rae/model/diffusion_loss/models/lightningDiT.py`: LightningDiT. PatchEmbed + fixed 2D
  sincos pos_embed; GaussianFourierEmbedding for t; ConditionEmbedder for y (the query reps);
  c = input_t + y (per-token), optional SiLU; stack of LightningDiTBlock (RMSNorm, NormAttention
  with RoPE+optional QK-norm, SwiGLU FFN, adaLN modulation -> shift/scale/gate or 4-way no-shift);
  LightningFinalLayer (adaLN -> shift/scale, linear) then unpatchify. use_DDT path appends the wide
  head (we DROP it). hidden>latent enforced via configs (DiT-0.5B hidden=1280, ..., 9.8B hidden=4096).
- `scale_rae/model/scale_rae_arch.py`: unified MetaQuery-style model. latent_queries =
  nn.Parameter(randn(vision_token_len, hidden)); encode_images via frozen vision tower; mm_projector
  MLP into LLM space (understanding) and a separate linear from LLM query outputs into DiT cond space
  (generation); flow-matching diffusion loss on the clean image features; bidirectional attention over
  image-token block.
- `scale_rae/model/multimodal_decoder/decoder.py`: GeneralDecoder = ViT (decoder_embed -> ViTMAELayer
  blocks -> decoder_pred -> unpatchify) mapping 256x1152 tokens to pixels. Recon losses live in the
  decoder training script, not this module.

## Code framework correspondence (scaffold <-> final)
Pre-method scaffold pieces (generic, no method names): a frozen image encoder; a text LLM; a block of
learnable conditioning tokens of unknown purpose; a connector MLP; a generic transformer denoiser with
adaLN conditioning whose body is TODO (no DDT, no width tricks named); a generic straight-path
flow-matching loss with a TODO timestep sampler (no shift named); a generic ViT pixel decoder with TODO
body; a two-optimizer training loop. Each TODO is filled by exactly one piece of the final code above.
