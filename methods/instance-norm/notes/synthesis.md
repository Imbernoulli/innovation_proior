# Instance Normalization synthesis (Phase 1.5)

arXiv 1607.08022 (verified). Ulyanov, Vedaldi, Lempitsky 2016. Canonical impl: DmitryUlyanov/texture_nets (InstanceNormalization.lua); modern = torch.nn.InstanceNorm2d.

## Setting / pain point
Feed-forward style transfer. Gatys 2016 = optimization-based: iteratively optimize an image to match style statistics (Gram matrices of shallow VGG features) + content statistics (deeper VGG features) -> great results but MINUTES per 512x512 image. Texture Nets (Ulyanov ICML 2016) and Johnson 2016 = learn a feed-forward generator g(x,z) that maps content image x (+ noise z) to stylized output in ONE pass, trained to minimize the same perceptual loss L(x_0, x_t, g(x_t,z_t)). Fast, but qualitatively WORSE than Gatys.

Observed problems with the feed-forward generators (these are the diagnostic findings):
1. Training on MORE examples gave WORSE qualitative results — a net trained on 16 images beat one trained on thousands. Best results required few images + early stopping. The objective seemed "too hard to learn."
2. Worst artifacts along the image BORDER, from zero-padding before each conv. Fancier padding didn't fix it.
Conjecture: the standard architecture struggles to learn the right function.

## The key observation (insight)
Stylization output should NOT depend on the CONTRAST of the content image. The style loss makes the stylized image's contrast match the STYLE image's contrast, essentially independent of the content's contrast (Fig: low-contrast content -> same stylized contrast). So the generator must DISCARD content contrast. Question: implement contrast normalization by composing standard CNN layers (conv+ReLU+pool), or bake it into the architecture?

A simple contrast normalization (per image, per channel, dividing by spatial sum):
  y_tijk = x_tijk / Σ_l Σ_m x_tilm.   (Eq 1, hard to express as conv+ReLU)

The generators already CONTAIN a normalization layer: batch normalization. Compare.

## The formulas (verified Eq 2,3; note source typo "mu" should read μ)
Tensor x ∈ R^{T×C×W×H}; t=batch index, i=channel, j,k=spatial.

BATCH NORM (Eq 2): normalizes per channel i across the WHOLE batch (all T images + all spatial):
  y_tijk = (x_tijk − μ_i)/√(σ_i² + ε)
  μ_i = (1/(HWT)) Σ_t Σ_l Σ_m x_tilm
  σ_i² = (1/(HWT)) Σ_t Σ_l Σ_m (x_tilm − μ_i)²

INSTANCE NORM (Eq 3): normalizes per (image t, channel i) across SPATIAL only (drop the batch sum):
  y_tijk = (x_tijk − μ_ti)/√(σ_ti² + ε)
  μ_ti = (1/(HW)) Σ_l Σ_m x_tilm
  σ_ti² = (1/(HW)) Σ_l Σ_m (x_tilm − μ_ti)²

The ONLY change vs BN: statistics computed per individual image (per t) instead of pooled over the batch. This subtracts each image's own mean and divides by its own spatial std per channel = exactly a (learnable, via affine) contrast normalization. So contrast-normalization (Eq 1's intent) IS bakeable into the architecture, as a tweak to BN.

## Two design decisions
1. Per-instance statistics (drop the batch dimension from the moments): removes instance-specific mean/contrast shift, which is exactly what the generator otherwise has to learn the hard way. Makes the objective easier; the content contrast is normalized away by construction.
2. Apply INSTANCE norm at TEST time TOO, with the SAME per-instance computation. Contrast with BN: BN at test time uses FROZEN running population statistics (and is often folded/simplified away). Instance norm has NO batch dependence and NO population statistics — its statistics are a deterministic function of the single input image — so it behaves identically at train and test. This is essential: at test you stylize one image, and you WANT its own contrast normalized, not some training-set average.

## Why BN is wrong here
BN normalizes a channel using statistics pooled across the batch, so a single image's contrast is only partially removed (it's mixed with the rest of the batch's), and at test time it's replaced by fixed training-set statistics — so content-specific contrast is NOT removed per image. The generator still has to learn to strip the residual contrast itself, which is the hard part. Instance norm strips each image's contrast directly and identically at train/test.

## Implementation (verified InstanceNormalization.lua / InstanceNorm2d)
- texture_nets trick: reshape input (N,C,H,W) -> (1, N*C, H, W) and apply SpatialBatchNormalization with N*C "channels". Then each of the N*C (image,channel) slices is normalized over its own (H,W) — which is exactly per-instance per-channel. Affine weight/bias (length C) repeated N times. eps default 1e-5.
- Output reshaped back to (N,C,H,W).
- Modern equivalent: nn.InstanceNorm2d(C, eps=1e-5, affine=True for the generator). Normalizes each sample's each channel over H,W; running stats not used (track_running_stats=False) so train==test.
- Replace BN with IN EVERYWHERE in the generator. Johnson's residual generator adopted for final results.

## Design-decision -> why
- Normalize per image, not per batch: discard the content image's contrast (which style transfer must ignore); removes instance-specific mean/contrast shift -> easier learning.
- Over spatial (H,W) only, per channel: contrast is a per-channel spatial-scale property; matches Eq 1 contrast norm.
- Keep affine γ,β per channel: let the network rescale after normalization (as in BN).
- Same op at test time (no running stats): single-image inference must normalize the actual input's contrast; no batch to pool over, deterministic function of one image -> train/test consistency.
- eps for numerical stability of the division.

## Eval settings (pre-method facts)
Feed-forward style transfer generators (Texture Nets architecture; Johnson residual architecture). Trained on content images (e.g. subsets / ImageNet-scale), fixed style image, perceptual loss via pretrained VGG (Gram-matrix style loss from shallow layers + content loss from deep layers). Qualitative comparison vs Gatys optimization and vs BN generators. Resolutions 256/512/1080.

## Unsourced: none. Formulas from main.tex Eq 1-3; impl from InstanceNormalization.lua. (Source has a LaTeX typo writing "mu" for μ inside the variance sums — clearly the mean.)
