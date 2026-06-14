## Research question

A KL-regularized autoencoder (`AutoencoderKL` from `diffusers`) is trained on CIFAR-10 32×32 images,
and the single thing being designed is the **training loss**. Architecture, optimizer, LR schedule,
mixed precision, gradient clipping, and EMA are all fixed; the only contribution allowed is the loss
function. The objective is the best reconstruction quality on the held-out test set, scored primarily
by reconstruction FID (rFID, lower is better), with PSNR and SSIM as supporting diagnostics. The loss
sees a reconstruction, a target, the encoder's diagonal-Gaussian posterior, and the current training
step — and must return a scalar to backpropagate plus a metrics dict. Everything below the editable
region is frozen.

## Prior art before the first rung (reconstruction-objective lineage)

The first rung — a pixel reconstruction term plus a KL regularizer — is the resolution of a line of
generative-autoencoder objectives. These precede the ladder; the editable contract below is the slot
they each fill.

- **Plain autoencoder (Hinton & Salakhutdinov 2006).** Encode to a bottleneck, decode, minimize pixel
  reconstruction error. Compresses, but the latent space has no probabilistic structure — no prior, no
  way to sample — and reconstruction alone is known not to learn a well-behaved code. Gap: no
  generative model, no regularized latent.
- **Denoising / sparse autoencoders (Vincent et al. 2008; Ranzato et al. 2007).** Add a hand-set
  corruption or sparsity penalty to force a useful code. The regularizer works but is ad-hoc — a knob
  bolted on, not derived from a likelihood. Gap: the regularizer is hand-designed, not principled.
- **Variational Bayes / mean-field (Jordan et al. 1999).** Bound an intractable likelihood with a
  tractable `q`, but coordinate-ascent updates need analytic expectations under conjugate families —
  hopeless once the decoder is a neural network — and fit a separate `q` per datapoint, which does not
  scale to a large image set. Gap: no neural likelihoods, no amortization.
- **Pixel reconstruction distances (L2 / L1).** Squared error is the negative log-likelihood of a
  fixed-variance Gaussian decoder; absolute error is the Laplace sibling, which penalizes large
  residuals less and tends to come out sharper. Both treat pixels as independent, so they reward the
  blurry conditional mean under reconstruction uncertainty. Gap: pixel independence rewards blur.

## The fixed substrate

The training loop in `custom_train.py` is frozen and must not be touched. It builds an `AutoencoderKL`
(3 blocks, 2 downsample stages, latent 8×8 at compression `f=4`; channel widths and latent channels
scale with `BLOCK_OUT_CHANNELS`/`LATENT_CHANNELS` across the small/medium/large scales), wraps it so a
forward pass does encode → sample (one reparameterized draw `z = mean + std·eps` inside
`posterior.sample()`) → decode, and runs AdamW (lr 4e-4, wd 1e-4), 5% warmup + cosine LR, autocast +
GradScaler, grad clip 1.0, and EMA at 0.999. It evaluates rFID/PSNR/SSIM on the 10,000-image test set.

One detail of the loop is load-bearing for the third rung: it has **built-in GAN support** that
activates only if the loss module exposes a discriminator. If the criterion has `disc` and `disc_opt`
attributes, then after a `disc_start` warm-up the loop, every step, (1) adds an adversarial generator
loss `g_loss = -mean disc(recon)` to the loss with an **adaptive weight** that balances the GAN
gradient against a reference gradient at the decoder's last layer `vae.decoder.conv_out.weight` — using
`criterion._perceptual_loss` as the reference if the module stored it — and (2) updates the
discriminator with a **hinge loss plus an R1 gradient penalty** (`gp_weight=10`) via the module's own
`disc_opt`. So the adversarial machinery lives in the *fixed loop*; a GAN-style loss module only has to
build the discriminator, set `disc_start`, and stash `_perceptual_loss`.

## The editable interface

Exactly one region is editable — the `VAELoss` class (lines 32–76 of `custom_train.py`). Every method
on the ladder is a fill of this same contract:

```python
class VAELoss(nn.Module):
    def __init__(self, device): ...
    def forward(self, recon, target, posterior, step):
        # recon:     [B, 3, 32, 32] reconstruction in [-1, 1]
        # target:    [B, 3, 32, 32] original     in [-1, 1]
        # posterior: DiagonalGaussianDistribution — posterior.kl() is the per-sample KL
        #            0.5*sum_j (mu_j^2 + sigma_j^2 - 1 - log sigma_j^2); also .mean, .logvar
        # step:      current training step (int)
        # returns:   (loss_tensor, metrics_dict)
        ...
```

Available inside the loss: `torch`, `torch.nn`, `torch.nn.functional`, `torch.fft`, `lpips`, `numpy`,
`math`. The reconstruction sample is already baked into `recon` (one reparameterized draw), so the loss
must **not** resample. A GAN-style fill may additionally define a discriminator module, a `disc_opt`,
and a `disc_start` step, and store `self._perceptual_loss` — the fixed loop reads those.

The starting point is the scaffold default: the loss is unimplemented. Each rung replaces exactly this
class (and, for the adversarial rung, adds a discriminator class beside it) and nothing else.

```python
# EDITABLE region of custom_train.py (lines 32-76) — scaffold default (unimplemented)
class VAELoss(nn.Module):
    """VAE training loss function.

    The loss receives:
        recon:     Reconstructed images [B, 3, 32, 32] in [-1, 1]
        target:    Original images      [B, 3, 32, 32] in [-1, 1]
        posterior: DiagonalGaussianDistribution from the encoder
                   - posterior.kl()   -> KL divergence [B, ...]
                   - posterior.mean   -> latent mean
                   - posterior.logvar -> latent log-variance
        step:      Current training step (int)

    Must return: (total_loss, metrics_dict)
    Available imports: torch, torch.nn, torch.nn.functional, numpy, lpips, torch.fft
    """

    def __init__(self, device):
        super().__init__()
        raise NotImplementedError("Implement VAELoss.__init__")

    def forward(self, recon, target, posterior, step):
        raise NotImplementedError("Implement VAELoss.forward")
```

## Evaluation settings

Three training scales, each a different model width and latent channel count over a fixed step budget:
**small** (`BLOCK_OUT_CHANNELS=(64,128,256)`, `LATENT_CHANNELS=4`, 20,000 steps), **medium**
(`(96,192,384)`, `LATENT_CHANNELS=8`, 30,000 steps), and **large** (`(128,256,512)`,
`LATENT_CHANNELS=16`, 30,000 steps), all on a single seed (42). Reconstruction quality is measured on
the full CIFAR-10 test set (10,000 images). The primary metric is **best rFID per scale** (lower is
better); the task score is the geometric mean of best rFID across the three scales. PSNR (dB, higher
better) and SSIM (higher better) are supporting diagnostics. The contribution is the loss design only.
