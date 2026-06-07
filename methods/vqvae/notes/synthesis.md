# VQ-VAE — Synthesis (Phase 1.5)

## Pain point / research question
Unsupervised representation learning that keeps the *important* (high-level, multi-dimensional) features of data in the latent space, while still optimizing maximum likelihood. Two coupled problems with the dominant VAE recipe:

1. **Posterior collapse.** Pair a VAE with a powerful autoregressive decoder (PixelCNN/WaveNet) and the latents get ignored — the KL term `KL(q(z|x)||p(z))` is minimized by making `q(z|x)≈p(z)` (latents carry no info) while the strong decoder models `p(x)` on its own. The variational-lossy-autoencoder analysis (Chen et al. 2016) makes this explicit: with a strong decoder the model prefers to push information out of the latent.
2. **Discrete latents are hard to train.** Language/speech are inherently discrete, but discrete latent VAEs (NVIL, VIMCO, Gumbel-softmax) either have high-variance score-function gradients or are biased, and none had closed the likelihood gap with continuous (Gaussian-reparameterization) VAEs.

Goal: a discrete-latent generative model that (a) trains with low-variance gradients like the Gaussian reparameterization trick, (b) does not collapse even with a powerful decoder, (c) matches continuous VAEs in likelihood.

## The core method (as landed)
- Codebook (embedding table) `e ∈ R^{K×D}`: K codes, each D-dim.
- Encoder produces `z_e(x) ∈ R^D` (per spatial location).
- Quantize by nearest neighbor: `k = argmin_j ||z_e(x) - e_j||₂`, then `z_q(x) = e_k`.
- Posterior `q(z=k|x)` is one-hot deterministic (1 for the argmin code, 0 else).
- Decoder reconstructs from `z_q(x)`.
- Prior `p(z)` uniform during VQ-VAE training → `KL = log K`, **constant**.
- After training, fit an autoregressive prior (PixelCNN over latent grid for images, WaveNet for audio) over the discrete codes for generation.

## Derivations to live out in reasoning.md

### ELBO with a deterministic one-hot posterior
`log p(x) ≥ E_{q(z|x)}[log p(x|z)] − KL(q(z|x)||p(z))`.
With `q` one-hot (deterministic) the expectation collapses to the single term `log p(x|z_q(x))`. With uniform prior `p(z)=1/K`:
`KL(q||p) = Σ_z q(z)log(q(z)/p(z)) = 1·log(1/(1/K)) = log K`, constant w.r.t. encoder params. So it drops from the gradient. **This is the structural reason posterior collapse cannot happen the usual way**: there is no KL pressure pulling q toward p (KL is fixed at log K, independent of x), so the latent channel is never "switched off" by the objective — to lower reconstruction loss the decoder *must* use z_q.

### Non-differentiability and the straight-through estimator
`z_q = e_{argmin}` — argmin has zero gradient a.e. (piecewise constant), so `∂z_q/∂z_e = 0`. Reconstruction loss gives no gradient to the encoder. Fix (Bengio et al. 2013 straight-through idea): on the backward pass, **copy** the decoder-input gradient straight to the encoder output, i.e. pretend `∂z_q/∂z_e = I`. Implementation trick:
`z_q = z_e + sg[e_k − z_e]`.
Forward: equals `e_k` (the `sg` is identity in forward). Backward: `sg[...]` has zero grad, so `∇z_q = ∇z_e` — gradient flows around the quantizer unchanged. Bias: the gradient is computed at `z_q` but applied to `z_e`; this is a biased estimator (the two differ by the quantization residual), but it is low-variance and worked.

### Why the codebook needs its own loss term
Because of the straight-through copy, the embeddings `e` receive **no** gradient from the reconstruction loss (the gradient skips around them). So `e` must be trained separately. Simplest dictionary-learning: move each used code toward the encoder outputs assigned to it — `||sg[z_e] − e||₂²` (this is the VQ / online k-means objective; sg on z_e so only e moves).

### Why the commitment loss
The codebook loss only moves `e`. Nothing yet stops the encoder output `z_e` from drifting/growing without bound: the latent space is dimensionless (the decoder can rescale), so if `e` trains slower than the encoder, `z_e` can run away and bounce between codes. Add a commitment term `β||z_e − sg[e]||₂²` (sg on e so only the encoder moves), forcing the encoder to "commit" to (stay near) the code it picked. Robust to β∈[0.1,2.0]; use β=0.25.

### Full loss (eq 4) and the stop-gradient split
`L = log p(x|z_q(x)) + ||sg[z_e(x)] − e||₂² + β||z_e(x) − sg[e]||₂²`
- Term 1 (reconstruction): trains decoder (directly) + encoder (via straight-through). Embeddings get nothing here.
- Term 2 (codebook/VQ): trains embeddings only (sg blocks encoder).
- Term 3 (commitment): trains encoder only (sg blocks embeddings).
The two sg's *partition* responsibility for the single quantity `||z_e − e||²`: one direction updates the dictionary, the other updates the encoder; without the split, both would chase each other on the same term and the assignment dynamics would be unstable.
With N latents: terms 2,3 averaged over the N positions.

### EMA codebook update (appendix, alternative to term 2)
Term-2 is just k-means; the closed-form optimum for `e_i` given its assigned set `{z_{i,j}}` is the mean `e_i = (1/n_i)Σ_j z_{i,j}`. Can't do exact means on minibatches → online EMA:
`N_i ← γN_i + (1−γ)n_i` ; `m_i ← γm_i + (1−γ)Σ_j z_{i,j}` ; `e_i = m_i/N_i`, with γ=0.99.
Advantage: codebook update is independent of the optimizer used for encoder/decoder; trains faster. (Sonnet adds Laplace smoothing on N_i with ε to avoid dead codes / division by zero.)

### Likelihood / MAP argument
`log p(x) = log Σ_k p(x|z_k)p(z_k)`. Since decoder trained with `z=z_q(x)` (MAP), at convergence it puts mass only on `z_q`, so `log p(x) ≈ log p(x|z_q(x))p(z_q(x))`, and by Jensen `log p(x) ≥ log p(x|z_q(x))p(z_q(x))`.

### Why a learned (autoregressive) prior, fit *after*
During training the prior is held uniform (keeps KL constant, decouples representation learning from prior learning). For *generation* you need a real `p(z)` to ancestral-sample codes. The codes form a structured grid/sequence with strong dependencies → use an expressive autoregressive model: PixelCNN over the 2D latent grid (only spatial masking since 1 channel), WaveNet over 1D audio codes. Fitting it after training (not jointly) is the simple choice; joint training left as future work.

## Design-decision → why table
| Decision | Why this / why not the alternative |
|---|---|
| Discrete latents | inherent fit for language/speech; compress to high-level content; alternative (continuous VAE) wastes capacity on local noise |
| Nearest-neighbor quantization vs Gumbel-softmax/Concrete | Gumbel is a continuous relaxation: low-variance-but-biased early, high-variance-but-unbiased late; never closed the gap with Gaussian reparam. Hard NN + straight-through gives low-variance gradients like the reparam trick. |
| vs NVIL/VIMCO (score-function) | high variance, need variance reduction / many samples; still didn't match continuous VAEs |
| vs soft-to-hard annealed VQ (Agustsson 2017) | tried it; from scratch the decoder just inverts the continuous relaxation so no real quantization happens → must use a hard bottleneck |
| Straight-through estimator (copy grad) | argmin has zero gradient; ST gives a usable, low-variance (biased) encoder gradient; subgradient through quantization also possible but ST simpler and worked |
| Codebook loss term (VQ) | embeddings get no grad from recon due to ST → need separate dictionary learning; k-means move toward assigned encoder outputs |
| stop-gradient on z_e in codebook term | only the dictionary should move toward encoder; keeps term from also pulling the encoder |
| Commitment loss β-term | latent space dimensionless → z_e can grow/oscillate if e lags; force encoder to commit near chosen code |
| stop-gradient on e in commitment term | only encoder should move toward code, not vice-versa (that's the codebook term's job) |
| β = 0.25 | robust across [0.1,2.0]; scales relative to reconstruction loss magnitude |
| Uniform prior during training, KL=log K | makes KL constant → no posterior-collapse pressure; decouples prior from representation learning |
| EMA codebook (optional) | optimizer-independent dictionary update; faster; = online k-means with Laplace smoothing |
| Autoregressive prior fit after training | need real p(z) to sample; codes are strongly dependent → PixelCNN/WaveNet; fitting after = simpler than joint |
| distances via ||a||²−2a·e+||e||² | expand squared L2 for an efficient batched matmul rather than explicit pairwise diff |

## Canonical implementation (code/sonnet_vqvae.py — DeepMind Sonnet v1)
- `VectorQuantizer`: flattens to [-1, D]; `distances = sum(z²) − 2·z·W + sum(W²)`; `argmin` (= `argmax(-distances)`); one-hot; `quantized = embedding_lookup`.
- `e_latent_loss = mean((sg[quantized] − inputs)²)` (commitment), `q_latent_loss = mean((quantized − sg[inputs])²)` (codebook); `loss = q_latent_loss + commitment_cost·e_latent_loss`. NOTE: `commitment_cost` = β multiplies the **commitment** term, matching eq 4.
- Straight-through: `quantized = inputs + sg[quantized − inputs]`.
- `perplexity = exp(−Σ p log p)` over codebook usage (diagnostic; not in loss).
- `VectorQuantizerEMA`: same forward; codebook updated by EMA of cluster sizes `N` and sums `m` (=`matmul(flat_inputs, encodings)`), with Laplace smoothing `(N+ε)/(n+Kε)·n`; only the commitment term remains in `loss`.
- Embedding init: uniform_unit_scaling (non-EMA) / random_normal (EMA).
