# DALL-E synthesis (grounding notes)

arXiv 2102.12092 — "Zero-Shot Text-to-Image Generation", Ramesh, Pavlov, Goh, Gray, Voss, Radford, Chen, Sutskever (OpenAI), ICML 2021. Verified by title search.
Canonical impl: openai/DALL-E (dVAE encoder/decoder); grounds dVAE. Transformer is the sparse decoder-only (Child 2019).

## Pain point / research question
Text-to-image generation. Prior work (DRAW captions Mansimov 2015, GAN-based Reed 2016, StackGAN, AttnGAN, DM-GAN, DF-GAN) evaluated on SMALL datasets (MS-COCO, CUB-200), with custom multi-scale architectures, attention, auxiliary losses, extra conditioning. Samples still suffer object distortion, illogical placement, unnatural fg/bg blending. Question: is dataset size + model size the limiting factor? Hypothesis: a single large autoregressive transformer trained on massive data could be a flexible high-fidelity text-controlled generator, with image-to-image etc. as EMERGENT capabilities.

## Background / tools
- Autoregressive transformers (Vaswani 2017) scaled with compute/size/data -> impressive in text (GPT-2 Radford 2019), images (iGPT Chen 2020), audio (Jukebox Dhariwal 2020). Suggests: model text+image tokens as ONE autoregressive stream.
- PROBLEM modeling pixels directly: using pixels as image tokens needs inordinate memory for high-res; likelihood objectives prioritize SHORT-RANGE pixel dependencies (Salimans PixelCNN++ 2017) -> capacity wasted on high-freq detail not the low-freq structure that makes objects recognizable.
- Two-stage compression (VQ-VAE Oord 2017, VQ-VAE-2 Razavi 2019): compress image to discrete tokens first, then model the tokens. DALL-E borrows this.
- VAE / ELBO (Kingma 2013, Rezende 2014). beta-VAE (Higgins 2016): larger KL weight.
- Gumbel-softmax / Concrete (Jang 2016, Maddison 2016): reparameterizable relaxation of categorical for discrete latents.
- Straight-through (Bengio 2013) - used by VQ-VAE; DALL-E uses gumbel instead.

## THE METHOD - two stages

Goal: train transformer to autoregressively model text+image tokens as a single stream.
View whole procedure as maximizing the ELB on joint likelihood of images x, captions y, tokens z.
Factorization: p_{theta,psi}(x,y,z) = p_theta(x | y,z) p_psi(y,z). Lower bound (eq):
  ln p_{theta,psi}(x,y) >= E_{z ~ q_phi(z|x)} [ ln p_theta(x | y,z) - beta * D_KL(q_phi(y,z|x), p_psi(y,z)) ].
where q_phi = dVAE encoder's distribution over the 32x32 image tokens given image x; p_theta = dVAE decoder's distribution over RGB images given image tokens; p_psi = transformer's joint distribution over text+image tokens. (Assume y conditionally independent of x given z.) Bound only holds for beta=1 but they use beta>1.

### STAGE 1: Learn the visual codebook (dVAE), maximize ELB wrt phi, theta only (images alone).
- dVAE compresses 256x256x3 RGB -> 32x32 grid of image tokens, each one of K=8192 values. Context reduction factor 192 (= 256*256*3 / (32*32) = 196608/1024 = 192).
- Initial prior p_psi = UNIFORM categorical over K=8192 codebook vectors. q_phi = categorical parameterized by the 8192 logits at each spatial position of encoder's 32x32x8192 output.
- ELB hard: q_phi discrete -> no reparameterization gradient. VQ-VAE used online cluster assignment + straight-through. DALL-E instead uses GUMBEL-SOFTMAX RELAXATION: replace expectation over q_phi with one over q^tau_phi; relaxation tight as temperature tau -> 0.
- Reconstruction term ln p_theta uses the LOG-LAPLACE distribution (see below).
- Optimize relaxed ELB with Adam (AdamW actually), exponentially weighted iterate averaging.
- Stability tricks:
  - Anneal tau: 1 -> 1/16 over first 150000 updates (cosine; linear -> divergence). tau=1/16 closes gap between relaxed and true ELB.
  - 1x1 convs at end of encoder & beginning of decoder (reduce receptive field around the relaxation -> better generalization to true ELB).
  - Multiply outgoing activations from enc/dec resblocks by a small constant (stable init).
  - KL weight beta increased 0 -> 6.6 over first 5000 updates (cosine). beta=6.6 promotes BETTER codebook usage and SMALLER final reconstruction error (contrary to usual tradeoff; they speculate small beta -> relaxation noise reduces codebook usage early -> worse ELB).
  - Step size annealed 1e-4 -> 1.25e-6 over 1.2M updates.
- dVAE arch (official code): conv ResNet, bottleneck resblocks, mostly 3x3 convs, 1x1 on skip connections when channels change. First encoder conv 7x7; last encoder conv 1x1 producing 32x32x8192 logits. Both first & last decoder convs 1x1. Encoder MAX-pooling to downsample (better ELB than avg); decoder nearest-neighbor upsampling. n_hid=256, 4 groups, 2 blocks/group, 3 maxpools => /8 spatial (256->32). Bottleneck resblock: relu,3x3,relu,3x3,relu,3x3,relu,1x1 (n_hid=n_out//4); id_path 1x1 if channels change; out = id_path(x) + post_gain*res_path(x), post_gain=1/n_layers^2.
- AdamW beta1=0.9 beta2=0.999 eps=1e-8 wd=1e-4; EMA decay 0.999. Loss divided by 256*256*3 so KL weight effectively beta/192. 3M updates, total batch 512.

### LOGIT-LAPLACE distribution (appendix)
l1/l2 recon = Laplace/Gaussian for ln p_theta. Mismatch: pixel values bounded [0,1] but Laplace/Gaussian supported on all of R -> mass placed outside admissible range. Fix: apply sigmoid to a Laplace RV -> distribution on (0,1). pdf:
  f(x | mu, b) = 1/(2 b x (1-x)) exp( -|logit(x) - mu| / b ).   (logit-Laplace)
Use log of RHS as recon term. Decoder outputs SIX feature maps: first 3 = mu for RGB, last 3 = ln b. Before encoding, map pixels [0,255] -> (eps, 1-eps) via phi: x -> ((1-2eps)/255) x + eps, eps=0.1 (avoids x(1-x) numerical problems). To reconstruct: x_hat = phi^{-1}(sigmoid(mu)).

### STAGE 2: Learn the prior (transformer), maximize ELB wrt psi, fix phi, theta.
- p_psi = 12-billion-param sparse decoder-only transformer (Child 2019).
- Text: BPE-encode lowercased caption, <=256 tokens, vocab 16384 (10% BPE dropout in training). Image: 32x32=1024 tokens, vocab 8192, obtained by ARGMAX of dVAE encoder logits (no gumbel noise at stage 2).
- Concatenate text + image tokens -> single stream, model autoregressively.
- Decoder-only; each image token can attend to all text tokens in any of the 64 self-attention layers. Three self-attention mask types: text-to-text = standard causal; image-to-image = row, column, or convolutional attention mask. Single attention op for all three interactions (text->text, image->text, image->image) better than separate independently-normalized ops.
- Padding: learn a special padding token separately for each of 256 text positions (used when no text token), rather than -inf logits. (Higher val loss but better OOD captions.)
- Loss: cross-entropy for text & image tokens, each normalized by count of that kind in batch; weight text by 1/8, image by 7/8 (primarily interested in images).
- Embeddings: image vocab embedding summed with a ROW and COLUMN embedding (broadcasted). 64 attn layers, 62 heads, per-head size 64 => d_model = 62*64 = 3968.
- Attention masks: conv mask (11x11 kernel, wraparound) only in last (64th) layer; else for layer i in [1,63]: column mask if (i-2) mod 4 == 0 else row. First four: row, column, row, row.
- Training: AdamW beta1=0.9 beta2=0.96 eps=1e-8 wd=4.5e-2, grad clip norm 4 (warmup only). per-resblock gradient scaling (mixed precision), PowerSGD gradient compression rank 896 (~86%). 1024 V100, batch 1024, 430000 updates.

### SAMPLE GENERATION / reranking
Draw N samples from transformer, rerank with a pretrained contrastive model (CLIP, Radford 2021): score image-caption match, keep top k. N=512 default, temperature t=1. Language-guided search.

## Why each choice
- Two-stage (compress then model tokens) not pixels: pixels need too much memory; likelihood wastes capacity on short-range/high-freq detail. Discrete tokens at 32x32 give 192x shorter context capturing the recognizable structure.
- dVAE with K=8192 large vocab: mitigate info loss from 8x spatial downsample.
- Gumbel-softmax over straight-through/VQ: a reparameterizable relaxation -> can backprop the ELB through the discrete sampling; anneal tau to harden to true categorical. (VQ's straight-through is biased & needs codebook EMA; gumbel gives a smooth path.)
- Argmax (not sample) at stage 2: model is underparameterized (12B for 250M pairs), so no need for the sampling regularizer; deterministic tokens.
- Logit-Laplace recon: matches bounded pixel support; predicts mu and scale b.
- beta=6.6 (>1): better codebook usage -> lower recon error in the relaxed regime.
- Single transformer for text+image, decoder-only autoregressive: scale hypothesis; one model, emergent image-to-image.
- Sparse row/col/conv attention (Child 2019): 1024 image tokens, dense attention over 1280 positions x 64 layers too expensive; row/col factorize the 2D grid attention.
- Text 1/8, image 7/8 loss weight: care about image modeling.
- Contrastive reranking: trade compute for quality, language-guided.

NO unsourced facts. All numbers from main text + appendix (verified). d_model=3968 = 62*64. context factor 192 verified.
