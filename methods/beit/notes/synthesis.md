# BEiT synthesis (for Phase 2)

## Pain point / goal
- ViTs are data-hungry; want self-supervised pretraining on unlabeled images to fix it.
- BERT's masked-language-modeling (MLM) is the most successful SSL recipe in NLP: mask ~15% of *discrete word tokens*, predict them with a softmax over a fixed vocabulary. Want the analog for ViT.
- Problem porting to images: image patches are continuous pixels, NO pre-existing vocabulary. Can't put a softmax over "all possible patches."
- Straightforward alternative: regress raw pixels of masked patches (MSE). But pixel regression wastes capacity on short-range dependencies + high-frequency detail (noted by DALL-E; also ViT reports masked-patch pixel/3-bit-color prediction underperforms). It is the most literal BERT→CV translation and it works poorly.

## The key idea
- Need a DISCRETE target that abstracts away pixel detail and captures semantics → a learned visual codebook.
- Two views of each image: (1) image PATCHES (16x16) = input to ViT (preserve raw pixels). (2) VISUAL TOKENS = discrete codebook indices from a pretrained discrete VAE (dVAE = the DALL-E tokenizer), 14x14 grid, vocab |V|=8192.
- MIM: mask ~40% of patches (blockwise), replace with learnable mask embedding e_[M], feed corrupted sequence to ViT, predict the visual token at each masked position via softmax over codebook (cross-entropy classification).
- Objective: max sum over images E_M [ sum_{i in M} log p_MIM(z_i | x^M) ].

## dVAE / tokenizer
- tokenizer q_phi(z|x): pixels → discrete tokens via codebook. decoder p_psi(x|z): tokens → reconstructed image. Reconstruction objective E_{z~q_phi}[log p_psi(x|z)].
- Discrete latent → non-differentiable → Gumbel-softmax relaxation to train. Uniform prior on q_phi.
- BEiT directly uses public DALL-E tokenizer (vocab 8192, downsample factor 8, so 224/16... actually image tokens size = image_size//8; for 112 input → 14x14). NB: patches are 16x16 on 224 → 14x14; visual tokens 14x14 too — counts match.
- get_codebook_indices = argmax over encoder logits (one-point q_phi).

## VAE perspective (ELBO derivation)
- Want log p(x | x_tilde) where x_tilde is masked image. ELBO:
  log p(x|x_tilde) >= E_{z~q_phi(z|x)}[log p_psi(x|z)] - KL[ q_phi(z|x) || p_theta(z|x_tilde) ]
- Two-stage (like VQ-VAE / VQ-VAE-2):
  - Stage 1: learn tokenizer as dVAE — minimize -E_{z~q_phi}[log p_psi(x|z)] (reconstruction), uniform prior.
  - Stage 2: learn prior p_theta (the MIM network) with q_phi, p_psi fixed. Simplify q_phi to one-point dist at z_hat = argmax q_phi(z|x). Then ELBO reduces to:
    sum ( E_{z~q_phi}[log p_psi(x|z)]  +  log p_theta(z_hat | x_tilde) )
  - 2nd term = BEiT MIM objective. The KL collapses because q_phi is a delta and p_theta is the thing being learned → KL term turns into the log p_theta(z_hat|x_tilde) cross-entropy.

## Blockwise masking (Algorithm 1)
- Rather than random per-patch masking, mask contiguous BLOCKS.
- Loop: s = Rand(16, 0.4N - |M|) (block size, min 16 patches); r = Rand(0.3, 1/0.3) (aspect ratio); a=sqrt(s*r), b=sqrt(s/r); top-left (t,l) random; add block; until |M| > 0.4N.
- Code: min_num_patches=16 (paper) / 4 (repo default), aspect log-uniform between log(0.3) and log(1/0.3), 10 attempts per block, target ~75 of 196 patches = 40%.
- Why blockwise: relieves short-range dependency exploitation; ablation shows it helps, especially segmentation. Inspired by SpanBERT/n-gram masking.

## Backbone (ViT-Base)
- 12 layers, hidden 768, 12 heads, FFN 3072, patch 16x16, 224 input → 14x14=196 patches. Prepend [S]/cls token. Learnable 1D pos embeddings. vocab 8192.
- MIM head: Linear(768 -> 8192) = lm_head, softmax classifier; W_c in R^{|V| x D}, b_c.

## Init trick (stabilize Transformer)
- Random init in [-0.02, 0.02]. Then for l-th layer rescale output projection of attention and FFN by 1/sqrt(2l). (fix_init_weight: rescale attn.proj.weight and mlp.fc2.weight by 1/sqrt(2*layer_id).) Stabilizes large-scale pretraining.

## Fine-tuning
- Classification: avg-pool patch reps -> softmax linear. softmax(avg({h_i} W_c)), W_c in R^{D x C}.
- Segmentation: SETR-PUP task layer (deconv decoder), or UperNet in appendix.
- Intermediate fine-tuning: SSL pretrain -> finetune on data-rich intermediate (ImageNet) -> finetune downstream.

## Pretraining setup
- ImageNet-1K 1.2M images, no labels. 224x224, 14x14 patches. mask at most 75 patches (~40%).
- 800 epochs / 500k steps, batch 2k. AdamW beta(0.9,0.999), lr 1.5e-3, warmup 10 ep, cosine decay, wd 0.05, stochastic depth 0.1, no dropout. grad clip 3.0 (base).
- Augment: random resized crop, horizontal flip, color jitter.

## Ablations (motivating/diagnostic — only those about alternatives, NOT BEiT wins)
- Removing visual tokens (= recover raw pixels) hurts a lot — even below from-scratch. So discrete tokens are THE key ingredient.
- Removing blockwise (random masking) hurts, esp segmentation.
- Recovering 100% of visual tokens (not just masked) harms downstream.
- (These contrast alternatives; the "BEiT wins X points" numbers are out of scope for context/reasoning.)

## Baselines / ancestors to elaborate
- BERT (Devlin 2019): MLM on discrete word tokens, softmax over vocab; blockwise/span masking variants (SpanBERT, UniLMv2, T5).
- ViT (Dosovitskiy 2021): patchify image, standard Transformer; data-hungry; preliminary masked-patch prediction (3-bit mean color) underperforms.
- DALL-E dVAE (Ramesh 2021): discrete VAE image tokenizer, Gumbel-softmax, 8192 codebook, factor-8 downsample.
- VQ-VAE / VQ-VAE-2 (van den Oord 2017, Razavi 2019): two-stage discrete latent learning.
- iGPT (Chen 2020): k-means cluster RGB into 9-bit palette, then GPT/BERT on clustered tokens — loses pixel info (tokens are both input and output); huge params.
- VAE (Kingma 2014): ELBO.
- Gumbel-softmax (Jang 2017; Maddison 2017): differentiable discrete sampling.
- Contrastive / self-distillation SSL for ViT: MoCo v3, DINO, SwAV — alternative SSL paradigms (the "discriminative" strand).

## Code structure (for scaffold ↔ final correspondence)
- MaskingGenerator -> blockwise mask
- DataAugmentationForBEiT -> two views + mask per image
- dVAE.get_codebook_indices -> labels
- VisionTransformerForMaskedImageModeling -> patch_embed, mask_token replace, blocks, norm, lm_head; returns masked logits
- train_one_epoch -> labels=ids[mask], CrossEntropy on masked positions
