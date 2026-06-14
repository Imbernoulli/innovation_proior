**Problem.** Train a KL-regularized `AutoencoderKL` on CIFAR-10 for the best reconstruction (rFID),
with everything but the loss frozen. The floor is the simplest principled objective: a pixel
reconstruction term plus a KL regularizer — the negative ELBO.

**Key idea.** Maximize the variational lower bound `L = E_q[log p(x|z)] - D_KL(q(z|x) || N(0,I))`,
equivalently minimize its negation. The diagonal-Gaussian bottleneck gives the KL in closed form
(`posterior.kl()` = `0.5 Σ_j (mu_j² + sigma_j² - 1 - log sigma_j²)`); the reconstruction term is the
decoder negative log-likelihood. Take the **Laplace/L1** form (`F.l1_loss`) rather than the
Gaussian/L2 form: heavier tails penalize large residuals less and yield sharper reconstructions than
squared error, which hedges toward the blurry conditional mean. (Despite the rung's `l2` name, the
scaffold fill uses L1.)

**Why.** Pixel independence means even L1 only loosely tracks perceptual quality, but it is the correct
floor: it is the bare negative ELBO, and its measured softness (a lagging rFID despite decent PSNR/SSIM)
is exactly what motivates a perceptual term next.

**Hyperparameters.** `kl_weight = 1e-6` — small, because the task is reconstruction-driven; under the
mean reduction this is the *effective* KL coefficient (it absorbs the decoder variance and the per-pixel
averaging), and the KL is present only to keep the code near the prior. One reparameterized sample is
already baked into `recon`, so the loss does not resample.

```python
# EDITABLE region of custom_train.py (lines 32-76) — step 1: L1 reconstruction + KL
class VAELoss(nn.Module):
    """Basic VAE loss: L1 reconstruction + KL divergence."""

    def __init__(self, device):
        super().__init__()
        self.kl_weight = 1e-6

    def forward(self, recon, target, posterior, step):
        rec_loss = F.l1_loss(recon, target)          # Laplace-decoder NLL (sharper than L2)
        kl_loss = posterior.kl().mean()              # closed-form D_KL(q||N(0,I)), ADD it
        loss = rec_loss + self.kl_weight * kl_loss   # negative ELBO
        return loss, {
            "rec_loss": rec_loss.item(),
            "kl_loss": kl_loss.item(),
        }
```
