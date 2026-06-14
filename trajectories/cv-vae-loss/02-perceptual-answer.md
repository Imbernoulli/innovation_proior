**Problem.** The L1 + KL floor pays a blur tax that scales inversely with capacity (small-scale rFID
53.00 vs large 5.35): per-pixel L1 rewards the conditional-mean blur that rFID — a deep-feature
distribution distance — punishes. The training loss disagrees with the metric.

**Key idea.** Measure reconstruction error in **deep feature space**, where Euclidean distance is
emergently perceptual (early layers ≈ appearance, deep layers ≈ semantics) and blur is penalized. Add a
**LPIPS** perceptual term on top of the existing L1 + KL skeleton. The harness exposes `lpips` as a
distance to *consume*, not to learn — there are no 2AFC judgments or calibration optimizer here — so use
the pre-calibrated `lpips.LPIPS(net='vgg')` as a **frozen black-box ruler**: `eval()`,
`requires_grad_(False)` on every parameter, so the representation being exploited is never retrained.

**Why.** L1 is kept as a stability anchor (perceptual loss alone can leave high-frequency artifacts,
since many images map to similar deep features); LPIPS carries the sharpness L1 cannot see; KL stays the
same closed-form leash. Inputs are cast to float before the LPIPS call so the VGG trunk runs in the
precision it was trained in under the loop's autocast.

**Hyperparameters.** `perceptual_weight = 0.5` (balances the calibrated perceptual distance against the
pixel anchor); `kl_weight = 1e-6` (unchanged — the KL was never the problem). One reparameterized sample
is already in `recon`; the loss does not resample.

```python
# EDITABLE region of custom_train.py (lines 32-76) — step 2: L1 + LPIPS + KL
class VAELoss(nn.Module):
    """Perceptual VAE loss: L1 + LPIPS + KL."""

    def __init__(self, device):
        super().__init__()
        import lpips
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)   # frozen perceptual ruler
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)
        self.kl_weight = 1e-6
        self.perceptual_weight = 0.5

    def forward(self, recon, target, posterior, step):
        rec_loss = F.l1_loss(recon, target)                          # pixel anchor
        p_loss = self.lpips_fn(recon.float(), target.float()).mean() # deep-feature distance (float under autocast)
        kl_loss = posterior.kl().mean()
        loss = rec_loss + self.perceptual_weight * p_loss + self.kl_weight * kl_loss
        return loss, {
            "rec_loss": rec_loss.item(),
            "p_loss": p_loss.item(),
            "kl_loss": kl_loss.item(),
        }
```
