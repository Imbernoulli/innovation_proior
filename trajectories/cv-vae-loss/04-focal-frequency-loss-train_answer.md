The adversarial rung came in cleanly below perceptual, and the per-scale shape confirms the whole ladder's logic: small went 19.64 → 14.75, medium 10.69 → 7.13, large 3.84 → 3.38, and the gmean fell from 9.31 to 7.08. Three rungs of attacking *spatial* quality — pixel anchor, then deep-feature perceptual distance, then patch-adversarial realism plus feature matching — have each helped, and each later one helped less, which tells me the spatial story is near its floor. The discriminator and LPIPS together are squeezing out the last generic softness, and the curve is flattening. To go meaningfully below 7.08 I need a *different kind* of residual error, not a fourth spatial term piled on three. All three of my objectives live in the spatial domain: per-pixel L1 scores pixel energy, LPIPS scores deep-feature agreement, the discriminator and feature matching score local patch realism — and none has an *explicit* notion of which spatial *frequencies* the reconstruction is getting wrong. That matters, because a continuous VAE through an $f=4$ bottleneck has a structural frequency bias: it fits low-frequency, smooth content fast (most of the pixel energy, cheap under L1) and systematically struggles to reproduce high-frequency detail — fine edges, texture grain — which is a small fraction of the pixel energy but a large fraction of what rFID's feature extractor and a human eye register. The spatial terms patch this only indirectly; the surviving residual is, I suspect, *frequency-localized*: a systematic mismatch in particular spectral bands that no spatial loss is shaped to target.

I propose adding **Focal Frequency Loss** (Jiang, Dai, Wu & Loy, ICCV 2021) as a complement on top of the strongest baseline recipe — measuring reconstruction error directly on the spectrum. The 2D discrete Fourier transform gives, per image channel, a complex value $F(u,v)$ for every spatial frequency, carrying amplitude and phase. Two images match iff their full complex spectra match, so a distance between spectra is a legitimate reconstruction objective, and unlike the spatial terms it is *natively* organized by frequency, so it can target the bands the spatial losses keep missing. The natural per-component distance is the squared Euclidean distance between complex values, $|F_r(u,v) - F_f(u,v)|^2$, averaged over the spectrum.

The load-bearing design choice is that a plain, equally-weighted spectral distance would repeat the spatial loss's sin in a new guise. After three spatial rungs the low-frequency components are mostly correct; the components still wrong are a small minority of hard, mostly high-frequency ones. An unweighted sum spends its gradient polishing the already-good frequencies and barely touches the missing ones — exactly the spatial loss's behavior, relabeled. So the frequency loss must *adaptively concentrate on the hard frequencies*, and operationally a component is hard exactly when the current frequency error $|F_r - F_f|$ is large, a quantity I have for free every step. I therefore gate each component's distance by a weight matrix that is large where the current error is large,
$$w(u,v) = |F_r(u,v) - F_f(u,v)|^{\alpha},$$
normalized into $[0,1]$. Where the model already matches a frequency, $w \approx 0$ and it is ignored; where it is far off, $w$ is large and dominates; and the spotlight is *dynamic* — as a once-hard frequency gets matched its weight decays and the loss moves to the next-hardest band. This is a self-generated curriculum over frequencies, with no hand-set band selection. The full loss is
$$\mathrm{FFL} = \frac{1}{N}\sum_{u,v} w(u,v)\,|F_r(u,v) - F_f(u,v)|^2.$$

The subtlety I must get right is that $w$ has to be a *constant* with respect to the parameters — detached, stop-gradient. If gradients flowed through $w = \text{distance}^\alpha$, the optimizer would see a perverse second-order incentive to manipulate the importance map rather than reduce the distance, tangling the focusing intent with the optimization. Detached, the gradient is exactly $w(u,v) \cdot \nabla(\text{distance}^2)$: the ordinary spectral-distance gradient reweighted by a fixed per-frequency spotlight that says "spend your effort here." Two normalizations keep the spotlight stable — divide the weight matrix by its per-image max so the loss scale does not silently drift as absolute errors shrink over training, clamp to $[0,1]$, and zero any NaN from the $0^\alpha$ corner. The default focusing strength is $\alpha = 1$, the robust starting point where the weight scales linearly with error magnitude ($\alpha = 0$ would recover the unweighted loss I just argued against). For 32×32 CIFAR a single global FFT (`patch_factor = 1`) is right — patch-wise spectra are for larger images — and the FFT uses orthonormal normalization so the transform is unitary and the spectral distance lands on the same scale as a spatial one, which matters because I am adding it to spatial terms.

The crucial composition point is that FFL is a *complement*, not a replacement. It constrains the spectrum but says nothing about absolute pixel values being right — one can in principle match the spectrum while drifting the spatial content — so it must ride on top of a spatial reconstruction term that anchors the pixels. The natural place is the strongest recipe I have, the vqgan rung: I keep L1 + LPIPS + KL + feature matching and the loop's adversarial machinery exactly as they are, and add $w_{\text{freq}} \cdot \mathrm{FFL}(\text{recon}, \text{target})$ on top. FFL's job is narrow and orthogonal — drag the output's frequency statistics toward the target's, concentrating on the bands the three spatial rungs keep missing — so it should *stack*, recovering frequency fidelity the spatial terms structurally cannot, without disturbing what they do well. I add the FFL term *always*, not gated on the discriminator warm-up, because the frequency mismatch is present from step one, and weight it at $w_{\text{freq}} = 1.0$ so the orthonormal-scaled spectral term is comparable to the L1 and perceptual terms. Everything else is carried over verbatim: the frozen VGG LPIPS at $w_p = 0.5$, the spectral-norm `NLayerDiscriminatorWithFeatures`, `disc_start = 5000`, stashing `_perceptual_loss` for the loop's adaptive weight, $\beta = 10^{-6}$, $w_{\text{fm}} = 1.0$, and the discriminator's `Adam(lr=1e-4, betas=(0.5, 0.9))`. The architecture stays continuous (no codebook, no quantizer), the adversarial logic stays in the fixed loop, and the only new line in the objective is the frequency term. I expect the small scale's 14.75 to fall hardest — the narrowest model through the tightest bottleneck loses the most high-frequency detail, so an explicit spectral term has the most band-mismatch to recover there — with medium and large ticking down by less, and the gmean clearing 7.08. PSNR and SSIM should hold or improve slightly, since matching the spectrum more faithfully restores high-frequency content that, unlike the perceptual-vs-pixel trade, is not in tension with pixel accuracy.

```python
# EDITABLE region of custom_train.py (lines 32-76) — finale: vqgan recipe + Focal Frequency Loss
class FocalFrequencyLoss(nn.Module):
    """Focal frequency loss (ICCV 2021): spectral distance gated by a detached
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
