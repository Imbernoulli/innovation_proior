**Problem.** Three spatial rungs (pixel L1, LPIPS, patch-adversarial + feature matching) have flattened
the curve to gmean rFID 7.08, with diminishing per-rung returns: all three measure error in the spatial
domain and none targets the frequency bands a bottlenecked autoencoder structurally loses. The surviving
residual is frequency-localized — invisible to spatial losses.

**Key idea.** Add a **Focal Frequency Loss** as a
*complement* on top of the strongest baseline recipe. Take the orthonormal 2D FFT of reconstruction and
target; the per-component distance is the squared Euclidean distance between complex values; gate each
component by a **detached, [0,1]-normalized hard-frequency spotlight** `w(u,v) = |F_r − F_f|^α` (α=1),
large where the current frequency error is large. The loss is `mean(w · |F_r − F_f|²)` — a
self-generated curriculum that concentrates gradient on the bands the spatial terms keep missing and
decays each band's weight as it is matched. It constrains the spectrum (the spatial terms anchor the
pixels), so it stacks: `total = L1 + 0.5·LPIPS + 1e-6·KL + 1.0·feat_match + 1.0·FFL`.

**Why FFL is well-posed.** Orthonormal `fft2` keeps the spectral scale comparable to a
spatial loss; the weight matrix is **detached** so the gradient is exactly `w·∇(distance²)` and the
optimizer cannot game the importance map; per-image max-normalization + clamp `[0,1]` + NaN→0 keeps the
spotlight stable; `patch_factor=1` is a single global FFT (correct for 32×32).

**Why it should clear the bar.** A continuous VAE through the `f=4` bottleneck has a structural frequency
bias (fits low frequencies, loses high-frequency detail); none of the three spatial terms targets the
spectrum explicitly, so the frequency residual is a genuinely new axis. Expected: largest gain on the
small scale (most high-frequency loss), gmean below 7.08, PSNR/SSIM held or improved (spectral fidelity
is not in tension with pixel accuracy).

**Hyperparameters.** FFL `loss_weight = 1.0`, `alpha = 1.0`, `patch_factor = 1`. Carried over from the
strongest baseline: `perceptual_weight = 0.5`, `kl_weight = 1e-6`, `feat_match_weight = 1.0`,
`disc_start = 5000`, discriminator `Adam(lr=1e-4, betas=(0.5, 0.9))`. The FFL term is added every step
(the frequency mismatch is present from the start, independent of the discriminator warm-up).

```python
# EDITABLE region of custom_train.py (lines 32-76) — finale: vqgan recipe + Focal Frequency Loss
class FocalFrequencyLoss(nn.Module):
    """Focal frequency loss: spectral distance gated by a detached
    hard-frequency spotlight. Complement to spatial losses."""

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1,
                 ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        pf = self.patch_factor
        _, _, h, w = x.shape
        assert h % pf == 0 and w % pf == 0, "patch_factor must divide H and W"
        ph, pw = h // pf, w // pf
        patches = [x[:, :, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
                   for i in range(pf) for j in range(pf)]
        y = torch.stack(patches, 1)
        freq = torch.fft.fft2(y, norm='ortho')              # orthonormal 2D FFT
        return torch.stack([freq.real, freq.imag], -1)

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            tmp = (recon_freq - real_freq) ** 2
            tmp = torch.sqrt(tmp[..., 0] + tmp[..., 1]) ** self.alpha   # |F_r - F_f|^alpha
            if self.log_matrix:
                tmp = torch.log(tmp + 1.0)
            if self.batch_matrix:
                tmp = tmp / tmp.max()
            else:
                tmp = tmp / tmp.max(-1).values.max(-1).values[:, :, :, None, None]
            tmp[torch.isnan(tmp)] = 0.0
            tmp = torch.clamp(tmp, min=0.0, max=1.0)
            weight_matrix = tmp.clone().detach()            # spotlight: detached constant gate

        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]           # |F_r - F_f|^2
        return torch.mean(weight_matrix * freq_distance)

    def forward(self, pred, target, matrix=None):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight


class VAELoss(nn.Module):
    """VQGAN-style spatial recipe + focal frequency loss complement."""

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
        self.freq_weight = 1.0

        self.ffl = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0, patch_factor=1)

        self.disc = NLayerDiscriminatorWithFeatures().to(device)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.disc_start = 5000

    def forward(self, recon, target, posterior, step):
        rec_loss = F.l1_loss(recon, target)
        p_loss = self.lpips_fn(recon.float(), target.float()).mean()
        self._perceptual_loss = p_loss            # loop's adaptive GAN weight balances against this
        kl_loss = posterior.kl().mean()
        freq_loss = self.ffl(recon.float(), target.float())   # frequency-domain complement

        disc_factor = 1.0 if step >= self.disc_start else 0.0
        feat_match_loss = 0.0
        if disc_factor > 0:
            _, real_feats = self.disc(target, return_features=True)
            _, fake_feats = self.disc(recon, return_features=True)
            for real_f, fake_f in zip(real_feats, fake_feats):
                feat_match_loss += F.l1_loss(fake_f, real_f.detach())
            feat_match_loss = feat_match_loss / len(real_feats)

        total_rec_loss = rec_loss + self.perceptual_weight * p_loss + self.freq_weight * freq_loss
        if disc_factor > 0:
            total_rec_loss = total_rec_loss + self.feat_match_weight * feat_match_loss

        loss = total_rec_loss + self.kl_weight * kl_loss

        metrics = {
            "rec_loss": rec_loss.item(),
            "p_loss": p_loss.item(),
            "kl_loss": kl_loss.item(),
            "freq_loss": freq_loss.item(),
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
