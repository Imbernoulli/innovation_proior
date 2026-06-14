**Problem.** The perceptual rung leaves a residual softness a static LPIPS ruler tolerates (small-scale
rFID stalls at 19.64, still the worst of the three): a fixed perceptual distance has no adversarial
opinion on whether a patch of texture is realistic.

**Key idea.** Add a **patch discriminator** that actively hunts the residual softness and forces the
autoencoder to produce realistic local texture — keeping only the *loss-only* idea from the VQ lineage.
This task is a fixed continuous `AutoencoderKL` (diagonal-Gaussian bottleneck): **no codebook, no
quantizer, no straight-through, no second-stage transformer** — none of that machinery exists here, and
none is imported. The adversarial logic lives in the **fixed training loop**: it computes the generator
logit loss `-mean disc(recon)`, runs the discriminator update with a **hinge loss plus an R1 gradient
penalty** (`gp_weight=10`), and sets an **adaptive weight** balancing the GAN gradient against
`criterion._perceptual_loss` at the decoder's last layer. The loss module only builds the discriminator,
sets `disc_start`, stashes `_perceptual_loss`, and adds the one term the loop does not: **feature
matching**.

**Why each piece.** PatchGAN concentrates the adversarial signal on local texture and gives a dense
gradient; spectral norm on every conv keeps `D` Lipschitz under the loop's every-step updates; feature
matching (L1 between real and fake discriminator features, real detached) is a denser, more stable
signal than the logit alone; `disc_start` warm-up holds adversarial pressure off until the autoencoder
reconstructs sanely; stashing `_perceptual_loss` makes the loop's adaptive weight balance against the
perceptual gradient (not the full loss). The L1 + LPIPS + KL skeleton is unchanged.

**Hyperparameters.** `perceptual_weight = 0.5`, `kl_weight = 1e-6`, `feat_match_weight = 1.0`,
`disc_start = 5000`; discriminator `Adam(lr=1e-4, betas=(0.5, 0.9))`. The feature-matching term is gated
on `step >= disc_start` so it activates with the loop's adversarial terms.

```python
# EDITABLE region of custom_train.py (lines 32-76) — step 3: L1 + LPIPS + KL + feature matching
class VAELoss(nn.Module):
    """VAE loss with adversarial training and feature matching."""

    def __init__(self, device):
        super().__init__()
        import lpips
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

        self.device = device
        self.perceptual_weight = 0.5
        self.kl_weight = 1e-6
        self.feat_match_weight = 1.0

        self.disc = NLayerDiscriminatorWithFeatures().to(device)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.disc_start = 5000

    def forward(self, recon, target, posterior, step):
        rec_loss = F.l1_loss(recon, target)
        p_loss = self.lpips_fn(recon.float(), target.float()).mean()
        self._perceptual_loss = p_loss            # the loop balances the adaptive GAN weight against this
        kl_loss = posterior.kl().mean()

        disc_factor = 1.0 if step >= self.disc_start else 0.0
        feat_match_loss = 0.0
        if disc_factor > 0:
            _, real_feats = self.disc(target, return_features=True)
            _, fake_feats = self.disc(recon, return_features=True)
            for real_f, fake_f in zip(real_feats, fake_feats):
                feat_match_loss += F.l1_loss(fake_f, real_f.detach())
            feat_match_loss = feat_match_loss / len(real_feats)

        total_rec_loss = rec_loss + self.perceptual_weight * p_loss
        if disc_factor > 0:
            total_rec_loss = total_rec_loss + self.feat_match_weight * feat_match_loss

        loss = total_rec_loss + self.kl_weight * kl_loss

        metrics = {
            "rec_loss": rec_loss.item(),
            "p_loss": p_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        if disc_factor > 0:
            metrics["feat_match"] = feat_match_loss.item()

        return loss, metrics


class NLayerDiscriminatorWithFeatures(nn.Module):
    """PatchGAN discriminator with intermediate feature extraction."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        from torch.nn.utils import spectral_norm
        self.n_layers = n_layers

        layers = []
        layers.append(spectral_norm(nn.Conv2d(input_nc, ndf, 4, 2, 1)))
        layers.append(nn.LeakyReLU(0.2, True))

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False)))
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, True))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, bias=False)))
        layers.append(nn.BatchNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, True))

        layers.append(spectral_norm(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)))

        self.model = nn.Sequential(*layers)
        self.features = []
        self._register_hooks()

    def _register_hooks(self):
        def hook(module, input, output):
            self.features.append(output)

        for layer in self.model:
            if isinstance(layer, nn.LeakyReLU):
                layer.register_forward_hook(hook)

    def forward(self, x, return_features=False):
        self.features.clear()
        out = self.model(x)
        if return_features:
            return out, self.features.copy()
        return out
```
