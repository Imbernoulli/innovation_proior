## Research question

Train an `AutoencoderKL` from `diffusers` on CIFAR-10 32×32 images. The only design choice is the **training loss**; architecture, optimizer, LR schedule, mixed precision, gradient clipping, and EMA are frozen. The objective is the best reconstruction quality on the held-out test set, measured primarily by reconstruction FID (rFID, lower is better), with PSNR and SSIM as diagnostics. The loss receives a reconstruction, a target, the encoder's diagonal-Gaussian posterior, and the current training step, and returns a scalar loss plus a metrics dict.

## Prior art / Background / Baselines

These objectives are the relevant reference points for the loss design.

- **Plain autoencoder.** Map input through an encoder-decoder bottleneck and minimize pixel reconstruction error.
- **Denoising / sparse autoencoders.** Augment reconstruction with a hand-set corruption process or sparsity penalty to shape the latent code.
- **Mean-field variational Bayes.** Approximate an intractable posterior with a tractable variational distribution and optimize a lower bound. Coordinate-ascent updates require analytic expectations under conjugate families.
- **Pixel reconstruction distances (L2 / L1).** Treat reconstruction as an independent Gaussian (L2) or Laplace (L1) likelihood over pixels.

## Fixed substrate / Code framework

The training loop in `custom_train.py` is frozen. It builds an `AutoencoderKL` (3 blocks, 2 downsample stages, latent 8×8 at compression `f=4`; channel widths and latent channels scale with `BLOCK_OUT_CHANNELS`/`LATENT_CHANNELS` across the small/medium/large scales), wraps encode → sample (`z = mean + std·eps` from `posterior.sample()`) → decode, and runs AdamW (lr 4e-4, wd 1e-4), 5% warmup + cosine LR, autocast + GradScaler, grad clip 1.0, and EMA at 0.999. It evaluates rFID/PSNR/SSIM on the full CIFAR-10 test set.

One load-bearing detail: the loop has **built-in GAN support** that activates only if the loss module exposes a discriminator. If the criterion has `disc` and `disc_opt` attributes, then after `disc_start` warmup the loop (1) adds an adversarial generator loss `g_loss = -mean disc(recon)` with an adaptive weight that balances the GAN gradient against a reference gradient at `vae.decoder.conv_out.weight` (using `criterion._perceptual_loss` if stored), and (2) updates the discriminator with hinge loss plus an R1 gradient penalty (`gp_weight=10`) via the module's own `disc_opt`. A GAN-style loss module only has to build the discriminator, set `disc_start`, and stash `_perceptual_loss`.

## Editable interface

Only one region is editable — the `VAELoss` class in `custom_train.py`. Every candidate loss fills the same contract:

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

Available inside the loss: `torch`, `torch.nn`, `torch.nn.functional`, `torch.fft`, `lpips`, `numpy`, `math`. The reconstruction sample is already baked into `recon` (one reparameterized draw), so the loss must **not** resample. A GAN-style fill may additionally define a discriminator module, a `disc_opt`, and a `disc_start` step, and store `self._perceptual_loss` — the fixed loop reads those.

The starting scaffold is unimplemented:

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

Three training scales, each with a different model width and latent channel count over a fixed step budget: **small** (`BLOCK_OUT_CHANNELS=(64,128,256)`, `LATENT_CHANNELS=4`, 20,000 steps), **medium** (`(96,192,384)`, `LATENT_CHANNELS=8`, 30,000 steps), and **large** (`(128,256,512)`, `LATENT_CHANNELS=16`, 30,000 steps), all on seed 42. Reconstruction quality is measured on the full CIFAR-10 test set (10,000 images). The primary metric is **best rFID per scale** (lower is better); the task score is the geometric mean of best rFID across the three scales. PSNR (dB, higher better) and SSIM (higher better) are supporting diagnostics.
