# VQGAN synthesis notes (for Phase 2)

## Pain point
- Transformers (Vaswani 2017) model long-range structure via full self-attention but cost is O(N^2) in sequence length. On images, N = H*W pixels, so resolution itself scales N quadratically -> infeasible past ~64x64 in pixel space (Parmar 2018 Image Transformer; Chen 2020 ImageGPT confined to <=192 via shallow VQ).
- CNNs cheap (local kernels, linear in pixels) but locality/weight-sharing bias is too restrictive for holistic/global composition; PixelCNN/PixelSNAIL convolutional AR density estimators can't capture long-range structure well at high res.
- Goal: high-res (megapixel) image synthesis, conditional & unconditional, with the expressivity of transformers but tractable compute.

## Key insight
Don't run the transformer on pixels. Compress the image with a CNN into a short grid of discrete codes (h x w << H x W), let the transformer model the prior over codes, decode codes back to pixels with a CNN. Split labor: CNN handles local perceptual detail + compression; transformer handles global composition over a short sequence.

## Stage 1 — VQGAN (the contribution over VQVAE)
- Base = VQVAE (van den Oord 2017): encoder E -> z_hat in R^{hxwxnz}; element-wise quantize each spatial vector to nearest codebook entry z_k in Z={z_k}_{k=1..K}; decode x_hat = G(z_q). Straight-through estimator copies decoder gradient past argmin: z_q = z_e + sg[z_q - z_e]. VQ loss = ||x - x_hat||^2 + ||sg[E(x)] - z_q||^2 + ||z_q - sg[E(x)]||^2 (codebook + commitment).
- Problem with VQVAE: L2 pixel reconstruction -> blurry. To push compression rate up (large f = downsampling factor) while keeping codebook short enough for a transformer, need *perceptually* faithful reconstruction at high compression. L2 averages -> blur.
- VQGAN changes the reconstruction objective:
  - Replace L2 rec term with a PERCEPTUAL loss: LPIPS (Zhang 2018) — distance in deep VGG feature space, matches human perception of texture/structure. (Impl: rec_loss = |x - x_hat| (L1) + perceptual_weight * LPIPS.)
  - Add a PATCH-based adversarial loss: a discriminator D (PatchGAN, Isola 2017 pix2pix) over patches distinguishing real vs reconstructed. L_GAN = [log D(x) + log(1 - D(x_hat))]. (Impl actually uses HINGE loss: D loss = 0.5*(mean relu(1-D(x)) + mean relu(1+D(x_hat))); G loss = -mean D(x_hat).)
  - Full objective: Q* = argmin_{E,G,Z} max_D E_x [ L_VQ + lambda * L_GAN ].
- ADAPTIVE GAN weight lambda (the load-bearing equation):
  lambda = grad_{G_L}[L_rec] / (grad_{G_L}[L_GAN] + delta)
  where G_L = last layer of decoder, grad denotes the L2 norm of the gradient of that loss w.r.t. the last layer's weights, delta = 1e-6 (paper) / 1e-4 (code) for stability.
  - Why: balances the two gradient signals so neither dominates regardless of their raw scales. Normalizes GAN gradient to the magnitude of the reconstruction gradient at the layer where they meet. Code clamps to [0, 1e4], detaches, multiplies by a base disc_weight.
- WARMUP: lambda set to 0 for an initial phase (>= 1 epoch) — train as pure autoencoder first so codebook/decoder are sane before adversarial pressure (disc_start threshold via adopt_weight).
- Single attention layer (non-local block) at lowest resolution in E and D to aggregate global context cheaply.
- beta on commitment loss: paper v3 changelog says beta was a bug (never applied; effectively 1.0); they removed it. (VQVAE used 0.25.) So L_VQ commitment weight = 1.0 in their runs.
- Architecture: E/D follow the DDPM (Ho 2020) UNet design without skip connections: Conv2D in, m x {ResnetBlock, Downsample}, ResBlock, NonLocal(Attn)Block, ResBlock, GroupNorm+Swish+Conv2D out -> nz channels. Decoder mirrors with Upsample. h = H/2^m, w = W/2^m, f = 2^m. Discriminator = NLayerDiscriminator (PatchGAN). 1x1 quant_conv to embed_dim and post_quant_conv back.
- Typical: |Z| = 1024, f = 16 (m=4), latent 16x16.

## Stage 2 — transformer prior
- After E,G fixed: image -> z_q -> sequence s of indices s_ij = k s.t. (z_q)_ij = z_k, length h*w. Raster (row-major) order.
- GPT-style decoder-only transformer (Radford 2019 GPT-2): p(s) = prod_i p(s_i | s_{<i}); loss = E[-log p(s)] = cross-entropy next-index.
- Conditional: p(s|c) = prod_i p(s_i | s_{<i}, c). If c spatial -> encode with its own VQGAN to indices r, PREPEND r to s (decoder-only conditioning). If c is a class label -> single index prepended (SOS-like).
- Ordering ablation: row-major best NLL (4.767) vs spiral/z-curve/subsample/alternate; AR code modeling is NOT permutation invariant; raster wins and matches sliding-window requirement.
- Sampling: transformer samples indices autoregressively (temperature t=1.0, top-k=100); map indices -> codebook entries -> G decodes to image.
- High-res / megapixel: latent grid still too long for full attention -> SLIDING-WINDOW attention. Crop to feasible latent patch during training; at sampling slide the attention window across the grid (row-major). Works because dataset stats approx spatially invariant or spatial conditioning available; else condition on coordinates (Lin 2019 COCO-GAN).
- GPT2-medium ~307M, block_size = h*w (+cond). Causal mask = lower-triangular.

## Code structure (maps to scaffold)
- VectorQuantizer: nearest-code, straight-through, codebook+commitment loss.
- Encoder/Decoder (diffusionmodules.model): ResnetBlock, Downsample/Upsample, AttnBlock (non-local), Normalize=GroupNorm, nonlinearity=Swish.
- VQLPIPSWithDiscriminator: LPIPS + L1 rec, NLayerDiscriminator (PatchGAN), hinge GAN, calculate_adaptive_weight (lambda), adopt_weight (warmup).
- VQModel (LightningModule): encode/decode, two optimizers (AE + disc), get_last_layer = decoder.conv_out.weight.
- GPT (mingpt): tok_emb + pos_emb, Blocks (CausalSelfAttention + MLP 4x GELU), causal mask, head -> vocab logits, cross_entropy.
- Net2NetTransformer (cond_transformer): encode_to_z, encode_to_c, prepend c indices, shift-by-one targets, sample (autoregressive top-k), decode_to_img.

## Design-decision -> why
- discrete codes (not continuous latent): transformer needs a finite vocabulary / categorical next-token; discrete also enables strong AR likelihood model. (from VQVAE.)
- nearest-neighbor VQ + straight-through: differentiable-ish discrete bottleneck (VQVAE).
- perceptual (LPIPS) instead of L2: L2 -> blur; need crisp texture at high compression.
- patch discriminator (not whole-image): texture realism is local; patch D gives dense gradient, fewer params, better high-freq detail (pix2pix rationale); also keeps codebook focused on perceptually important local structure so transformer needn't model low-level stats.
- adaptive lambda via gradient-norm ratio at last decoder layer: rec and GAN gradients differ wildly in scale and over training; fixed weight is brittle; normalize at the meeting point.
- warmup lambda=0: adversarial training on a random decoder/codebook destabilizes; get a decent autoencoder first.
- raster order: best empirical NLL + required by sliding window.
- sliding window: full attention over megapixel latent still infeasible; local window with spatial-invariance assumption.
- single low-res attention layer in AE: cheap global context without quadratic cost over full feature map.
