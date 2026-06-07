# ControlNet synthesis (Phase 1.5)

Verified: arXiv 2302.05543, "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet), Zhang, Rao, Agrawala (ICCV 2023).
Canonical impl: lllyasviel/ControlNet cldm/cldm.py (saved in code/). Built on Stable Diffusion (latent diffusion / LDM) U-Net.
NOTE on encoder: paper text says the conditioning encoder E = 4 conv layers, 4x4 kernels, 2x2 strides, ReLU, channels 16/32/64/128, Gaussian init, trained jointly. Canonical code input_hint_block uses 3x3 convs + SiLU, more layers, ending with a zero_module conv. I'll present the paper's stated encoder, noting it maps 512x512 image condition -> 64x64 feature matching SD latent size.

## Pain point / research question
- Text-to-image diffusion (Stable Diffusion) makes stunning images from prompts, but text gives poor SPATIAL control: precise layout, pose, shape, edges are hard/impossible to specify via text. Matching a mental image takes many trial-and-error prompt edits.
- Want: let users supply a conditioning IMAGE (edge map, depth, pose skeleton, segmentation, normals, scribble) that directly specifies spatial composition, and learn the condition->image control end-to-end.
- Challenge: condition-specific datasets are tiny (~100k) vs LAION-5B (~5B, 50,000x smaller) used to train SD. Direct finetuning / continued training of the big pretrained model on small data → OVERFITTING and CATASTROPHIC FORGETTING (it forgets its billions-of-images prior).
- So: how to add a new spatial control to a huge frozen text-to-image model, learning from small data, WITHOUT destroying the pretrained model's quality/capabilities?

## Core idea (insight-before-method)
1. LOCK (freeze) the production-ready pretrained model — never touch its weights → preserves its billions-of-images prior, no forgetting, and frees gradient/memory (no backprop through locked branch).
2. Make a TRAINABLE COPY of its encoding blocks → reuse the deep, robust pretrained encoder as a strong backbone for the new control, instead of training from scratch on tiny data.
3. The trainable copy takes the conditioning vector c as extra input.
4. Connect the copy back to the locked model via ZERO CONVOLUTIONS so that at init the whole thing equals the original model (adds zero), then grows the control parameters progressively from zero — no harmful noise corrupts the deep features at the start, protecting both the locked model AND the pretrained-initialized trainable copy.

## The ControlNet block (THE math)
For a trained neural block F(·;Θ): y = F(x;Θ).  (x,y ∈ R^{h×w×c}.)
Lock Θ; clone to trainable copy with params Θ_c. Connect with two zero convolutions Z(·;·):
Z = 1x1 conv, BOTH weight and bias initialized to ZERO. Params Θ_z1, Θ_z2.

ControlNet block output (eq:key1):
  y_c = F(x;Θ) + Z( F( x + Z(c;Θ_z1) ; Θ_c ) ; Θ_z2 ).
At first training step both Z(·)=0 (zero weight & bias):
  y_c = F(x;Θ) + Z(F(x+0;Θ_c);Θ_z2) = F(x;Θ) + 0 = y.    (eq key3)
So output identical to original block at init → no harmful noise. AND since Z(c;Θ_z1)=0 the trainable copy receives F(x;Θ_c) (gets the real x), so it is fully functional and retains pretrained capability — a strong backbone.

## Why zero-init layers can still train (the crucial supplementary gradient derivation — verified manually)
Consider a single zero conv as y = w⊙x + b (1x1 conv; treat as linear), with w=0, b=0 at init.
Partial derivatives: ∂y/∂w = x, ∂y/∂x = w, ∂y/∂b = 1.
- ∂L/∂w = (∂L/∂y)·x. The upstream gradient ∂L/∂y is NONZERO (loss flows back through the locked branch F(x;Θ) which produces a real nonzero y and a real loss). And x here is the trainable-copy output F(x+Z(c);Θ_c), which is NONZERO because the copy receives the real input x. So ∂L/∂w ≠ 0 even though w=0 — the gradient depends on x and the loss, NOT on w.
- ∂L/∂x = (∂L/∂y)·w = 0 at the first step (w=0): so the trainable copy + Z_z1 receive no gradient YET.
- ∂L/∂b = ∂L/∂y ≠ 0.
After ONE gradient step w becomes ≠ 0; then ∂L/∂x = w·(∂L/∂y) ≠ 0, so gradients now flow into the trainable copy and the first zero conv too. So the network bootstraps off zero: the outer zero conv's weight moves first, opening the path. (Different weights get different gradients since x_i, v_j vary across positions/channels — no symmetry lock.)
This produces the "sudden convergence phenomenon": output is always high quality (starts = original model), then at some step the model abruptly starts following the condition (<10k steps).

## Applying to Stable Diffusion (architecture)
- SD = U-Net (LDM, latent diffusion): encoder, middle block, skip-connected decoder. Encoder 12 blocks, decoder 12 blocks, +1 middle = 25 blocks total. 8 are up/down-sampling convs; 17 main blocks each = 4 resnet layers + 2 ViTs (cross-attn + self-attn). Text via CLIP text encoder; timestep via positional time encoder.
- ControlNet = trainable copy of the 12 encoding blocks + 1 middle block. 12 encoding blocks at 4 resolutions (64,32,16,8) × 3 each. Outputs added to the 12 U-Net decoder skip-connections + 1 middle block (via zero convs).
- Locked SD decoder & encoder: frozen, NO gradient → ~23% more GPU memory, ~34% more time per iter vs plain SD.
- Conditioning image encoder E: input condition image 512×512 → encode to 64×64 feature c_f matching SD latent size. Paper: 4 conv layers, 4×4 kernels, 2×2 strides, ReLU, channels 16/32/64/128, Gaussian init, trained jointly. c_f = E(c_i).
- SD works in LATENT space: 512×512 pixel images → 64×64 latents via a VQ-GAN-like encoder. (Rombach LDM; Esser VQGAN.)

## Training (grounded)
- Diffusion loss (same as SD, used directly for finetuning): L = E_{z0,t,c_t,c_f,ε~N(0,1)} [ ||ε - ε_θ(z_t, t, c_t, c_f)||_2^2 ].
  z_t = noisy latent at step t; c_t = text prompt; c_f = task condition.
- Randomly replace 50% of text prompts c_t with EMPTY strings during training → forces ControlNet to recognize semantics from the conditioning image itself (not lean on the prompt).
- Robust on small (<50k) and large (>1m) datasets; depth-to-image trainable on a single RTX 3090Ti.

## Inference extras (sec:infer)
- Classifier-free guidance + a "CFG Resolution Weighting" to balance.
- Compose MULTIPLE ControlNets (e.g. depth + pose): add their outputs to the SD directly.
- Can control how strongly conditions affect denoising.

## Baselines / prior art to elaborate (finetuning families it reacts to)
- Direct finetuning / continued training: overfits, forgets.
- HyperNetwork (Ha 2017): small net predicts weights of larger one; used to restyle SD.
- Adapter (Houlsby 2019; T2I-Adapter Mou 2023 concurrent): insert small new modules into frozen transformer; T2I-Adapter adapts SD to external conditions.
- Additive learning / Side-Tuning (Zhang 2020): frozen model + side branch, blend outputs with predefined weight schedule.
- LoRA (Hu 2021): low-rank weight offsets; avoids forgetting (low intrinsic dim).
- Zero-initialized layers: Nichol 2021 "zero_module" scales conv init to zero; ProGAN/StyleGAN/Noise2Noise manipulate init; Stability model cards use zero weights. (Gaussian init usually less risky than zero — arfin2020.)
- Stable Diffusion / LDM (Rombach 2021): latent diffusion U-Net base. VQGAN (Esser 2021) latent encoder. DDPM (Sohl-Dickstein 2015 / Ho). CLIP text encoder (Radford 2021).
- Image-to-image GANs (pix2pix Isola 2017, CycleGAN): condition→image mapping; trained from scratch / can't reuse a huge T2I prior. Palette (diffusion i2i from scratch), PITI (pretraining-based).
- Spatial control of diffusion: MakeAScene, SpaText, GLIGEN (new attention params), Textual Inversion / DreamBooth (personalize by finetuning).

## Evaluation settings (no outcomes)
- Conditions: Canny edges, Hough/M-LSD lines, HED edges, user scribbles, human pose, ADE20k segmentation, depth, normals, sketches/cartoon line.
- Base: Stable Diffusion v1.5 / v2.1 (same U-Net).
- Metrics: FID (distribution distance over 512×512 sets), CLIP text-image score, CLIP aesthetic score; user studies; ablations (replace zero conv with Gaussian conv; ControlNet-lite = single conv per block instead of trainable copy).

## Design-decision → why table
- Lock the pretrained model → preserve billions-of-images prior, no catastrophic forgetting, no gradient/memory in locked branch.
- Trainable COPY of encoder (not from scratch) → reuse robust pretrained features as backbone for small-data control.
- Zero convolution (1x1, weight+bias=0) connecting copy to locked model → at init output = original (no harmful noise); params grow from zero, protecting both branches; gradient on the outer zero-conv weight is nonzero (depends on x and loss, not w), so it bootstraps off zero.
- TWO zero convs (input c-side Z_z1 and output Z_z2) → input side lets copy initially ignore c (so it's fully functional = pretrained backbone); output side makes the added control start at zero.
- Apply to encoder + middle, add to decoder skips → U-Net skip structure; control injected where decoder reads features.
- Conditioning encoder E (512→64) → match SD latent resolution.
- 50% empty prompts → force the model to read the condition image, not rely on text.
- Diffusion ε-prediction L2 loss unchanged → finetune with the model's native objective; minimal change.
- CFG resolution weighting / multi-ControlNet composition by addition → inference-time control.
```
