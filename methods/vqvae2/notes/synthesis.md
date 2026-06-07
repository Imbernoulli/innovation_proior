# VQ-VAE-2 synthesis (grounding notes)

arXiv 1906.00446 — "Generating Diverse High-Fidelity Images with VQ-VAE-2", Razavi, van den Oord, Vinyals (DeepMind). Verified by title search.
Canonical impl: rosinality/vq-vae-2-pytorch (vqvae.py, train_vqvae.py) — ported from DeepMind Sonnet vqvae.py.

## Pain point / research question
Want HIGH-fidelity, HIGH-resolution, DIVERSE image generation. Two families:
- GANs (BigGAN, StyleGAN): great quality/resolution but mode collapse, poor diversity coverage, hard to evaluate (no test-set generalization measure; IS/FID are proxies).
- Likelihood models (VAE, flows, autoregressive PixelRNN/PixelCNN/WaveNet): cover all modes (MLE = forward KL, infinite penalty for missing a data mode), measurable NLL, but:
  - Pixel-space NLL is not a good proxy for sample quality (Theis 2016). No incentive to focus on global structure.
  - Autoregressive in pixel space is brutally slow to sample (sequential over ~256x256x3 pixels) and spends capacity modeling imperceptible local detail.

Idea (from lossy compression, JPEG removes >80% data invisibly): compress image to a small DISCRETE latent grid with an autoencoder (VQ-VAE), then model the prior over those codes with a powerful autoregressive net. Sampling/training in latent space is ~30x faster; autoregressive model freed from modeling negligible info.

## Prior art (load-bearing ancestors)
1. VQ-VAE (van den Oord, Vinyals, Kavukcuoglu 2017, 1711.00937) — THE foundation. Already fully documented in methods/vqvae.
   - Encoder E(x) -> continuous vector; quantize to nearest codebook prototype e_k, k=argmin_j ||E(x)-e_j||.
   - Straight-through: forward z_q=e_k, backward copy gradient to E(x). Implemented z_q = z_e + sg[e_k - z_e].
   - Loss eq (paper eq 2): L = ||x - D(e)||_2^2 + ||sg[E(x)] - e||_2^2 + beta||sg[e] - E(x)||_2^2.
     term1 recon, term2 codebook loss (moves code to encoder), term3 commitment (moves encoder to code), beta=0.25.
   - EMA codebook update (replaces term2): N_i^(t)=gamma N_i^(t-1)+(1-gamma) n_i^(t); m_i^(t)=gamma m_i^(t-1)+(1-gamma) sum E(x); e_i = m_i/N_i. gamma=0.99 default.
   - Deterministic posterior + fixed uniform prior => KL = log K constant => no posterior-collapse pressure; discrete bottleneck load-bearing.
2. PixelCNN / Gated PixelCNN (van den Oord 2016) — autoregressive p(x)=prod p(x_i|x_<i) via masked convs. Used as the prior over codes.
3. PixelSNAIL (Chen et al. 2017, 1712.09763) — gated residual PixelCNN blocks INTERLEAVED with causal multi-head self-attention. Attention gives unbounded receptive field for long-range structure that convs miss. SOTA NLL CIFAR 2.85, ImageNet32 3.80 bpd. VQ-VAE-2 top prior = this.
4. BigGAN (Brock et al. 2018) — SOTA FID/IS; introduced truncation trick to trade diversity for quality. Motivates VQ-VAE-2's classifier-based rejection sampling (an analogous quality/diversity knob for likelihood models).
5. VLAE / learned priors (Chen et al. 2016) — fitting a prior to the aggregate posterior reduces the marginal-posterior/prior gap; closer to true entropy => more coherent decoded samples. Justifies the post-hoc stage-2 prior.
6. Hierarchical latents (Rezende 2014); Dieleman 2018 (hierarchical VQ for music, WaveNet decoder, suffers hierarchy collapse). VQ-VAE-2 differs: levels extract COMPLEMENTARY info (each conditioned on pixels), feed-forward MSE decoder => no hierarchy collapse.

## The method (what's NEW vs VQ-VAE)
Two-stage:
STAGE 1 — Hierarchical VQ-VAE.
- Motivation: a single flat code can't both hold global shape/geometry and local texture for a large image; one codebook resolution forces a compromise. Split into a hierarchy:
  - TOP code (32x32 for 256px): global structure.
  - BOTTOM code (64x64): local detail, CONDITIONED on top code.
- Key design: bottom is conditioned on top, AND each level depends on pixels directly. If bottom only refined top, top must encode every detail. Letting each level see pixels encourages complementary info per level, reducing recon error. (Contrasts with Dieleman's pure-refinement hierarchy.)
- Encoder: enc_b downsamples x by 4 (256->64); enc_t downsamples enc_b by 2 (64->32). Quantize top -> e_top. Decode top, concat with enc_b, quantize -> e_bottom.
- Decoder: takes both levels; upsample top by 2, concat with bottom, decode by 4 back to 256.
- Same VQ loss & EMA, beta=0.25, codebook size K=512, codebook dim D=64, EMA gamma=0.99. Recon = MSE in pixels.
- Hyperparams (ImageNet-256 VQ-VAE): hidden 128, residual units 64, 2 res layers, enc conv filter 3, upsample filter 4.

STAGE 2 — Priors over the latent codes.
- TOP prior: PixelCNN (gated residual conv) with multi-head causal self-attention every five layers (because top is small 32x32 => attention affordable, and global structure needs long-range receptive field). Conditioned on class label. Plus deep 1x1-conv residual stack on top of PixelCNN output stack -> improves likelihood cheaply. Dropout after residual blocks + on attention logits.
- BOTTOM prior: conditional PixelCNN at 64x64. NO attention (too memory-expensive at this resolution; and local info doesn't need large receptive field, since conditioned on top). Uses a large/deep residual CONDITIONING stack fed from the top code. Conditioned on class label + top code.
- Train each prior separately (memory/compute), fit on the codes extracted from all data (Algorithm 2).
- Sampling: e_top ~ p_top; e_bottom ~ p_bottom(e_top); x = D(e_top, e_bottom). Ancestral.
- Top-prior hyperparams ImageNet256: hidden 512, residual 2048, 20 layers, 4 attention layers, 8 heads, filter 5, dropout 0.1, output stack 20 layers. Bottom: residual 1024, 20 layers, 0 attention, conditioning stack 20 blocks, dropout 0.1.

STAGE 3 (optional) — Classifier-based rejection sampling.
- MLE forces covering all modes -> some low-quality samples; ancestral sampling accumulates errors over long sequences.
- Use a pretrained ImageNet classifier; score each sample by prob it assigns to the intended class. Keep top-scoring fraction. Trades diversity for quality, analogous to BigGAN truncation but for likelihood models. (Critic/rejection = posterior method, "more on data manifold => classified correctly".)

## Design-decision -> why table
- Two-stage (autoencoder then prior), not end-to-end: keeps encoder/decoder feed-forward & fast; lets a slow autoregressive model live only in the compressed space (30x faster); decouples representation learning from density modeling.
- Discrete codes via VQ (not continuous VAE): natural fit for an autoregressive categorical prior (PixelCNN over indices); avoids posterior collapse (KL=const); compression.
- Hierarchy (top/bottom): separate global vs local so each prior captures the correlations at its scale; complementary not refining.
- Bottom conditioned on top + pixels: prevents top from having to carry all detail; complementary info reduces recon error.
- Attention only in top prior: 32x32 affordable, global structure needs long-range; 64x64 attention is memory-prohibitive and unnecessary (local + conditioned).
- EMA codebook (vs gradient codebook loss): from VQ-VAE; more stable dictionary learning, like online k-means; gamma=0.99.
- beta=0.25 commitment: from VQ-VAE; robust across 0.1-2.0.
- Learned (vs fixed uniform) prior in stage 2: closes marginal-posterior/prior gap => coherent samples; info-theoretically lossless re-encoding to bit rate near entropy.
- Rejection sampling: quality/diversity knob without retraining; counters error accumulation in ancestral sampling.
- MSE recon (feed-forward decoder, not autoregressive pixel decoder): avoids hierarchy-collapse problems; fast decoding.

## Code structure (rosinality), maps to scaffold
- class Quantize(dim, n_embed, decay=0.99, eps=1e-5): embed buffer [dim,n_embed]; cluster_size, embed_avg buffers. forward: flatten, dist = ||z||^2 - 2 z e + ||e||^2, argmin via (-dist).max; one-hot; quantize=embed_code; if training EMA update with Laplace smoothing (cluster_size+eps)/(n+n_embed*eps)*n; diff = (quantize.detach()-input)^2.mean() [commitment]; straight-through quantize = input + (quantize-input).detach().
- ResBlock: ReLU, conv3x3 in->ch, ReLU, conv1x1 ch->in, +input.
- Encoder(stride 4 or 2): strided conv4 downsamples, res blocks. Decoder: conv3, res blocks, ConvTranspose4 upsample.
- VQVAE: enc_b stride4, enc_t stride2, quantize_conv_t 1x1->embed_dim, quantize_t; dec_t stride2; concat[dec_t, enc_b]; quantize_conv_b; quantize_b; upsample_t ConvTranspose; dec stride4. forward returns (dec, diff). encode/decode/decode_code.
  defaults: channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512.
- train: criterion MSELoss; latent_loss_weight=0.25; loss = recon_loss + 0.25*latent_loss; Adam lr (3e-4 default), batch 128.

NO unsourced facts. All hyperparams from appendix tables / code.
