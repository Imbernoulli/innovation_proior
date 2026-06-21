The perceptual rung did what I bet it would. The small scale fell from 53.00 to 19.64 — the blur tax was real and LPIPS collected most of it, just where capacity is tightest — while medium went 15.75 → 10.69 and large 5.35 → 3.84, the diminishing-returns-up-the-capacity-ladder shape I expected. But look at where the small scale *stopped*: 19.64 is far better than 53.00 and still the worst of the three by a wide margin. There is a residual softness LPIPS could not finish off, and it is the one I flagged at the close of the last rung. A fixed perceptual distance pulls deep features together but has no sharp, *adversarial* opinion about whether a patch of texture is realistic — VGG features can be matched by something close-but-still-a-bit-soft, because LPIPS is a static ruler, not a critic that actively hunts for the residual tell-tale difference between a real image and a reconstruction. The 19.64 is what is left after the ruler has done all it can; closing it needs a signal that keeps finding the softness LPIPS tolerates and punishing it.

I propose adding a **patch discriminator with feature matching** — an adversarial loop on top of the L1 + LPIPS + KL skeleton. The name invites the wrong construction, so I am precise about what this rung is and is not. The lineage it descends from is a two-stage vector-quantized model: compress an image to a grid of discrete codes with a straight-through quantizer, swap the codebook's L2 objective for a perceptual-plus-adversarial one, then learn an autoregressive transformer prior over code indices. *None of that second machinery exists here.* The architecture is a fixed continuous `AutoencoderKL` with a diagonal-Gaussian bottleneck — no codebook, no quantizer, no straight-through estimator, no second-stage transformer prior. I am editing the *loss only*, so exactly one idea carries over: the reconstruction objective should be perceptual *plus patch-adversarial* rather than pixel only. Everything else from that lineage is out of scope.

The next thing the harness dictates is *where the adversarial machinery lives*, and this is load-bearing. The fixed training loop already has built-in GAN support: if my loss module exposes `disc` and `disc_opt` attributes, then after a `disc_start` warm-up the loop, every step, computes the generator adversarial loss $g_{\text{loss}} = -\mathbb{E}[\,\text{disc}(\text{recon})\,]$, weights it adaptively, adds it to my loss, and runs a separate discriminator update through my `disc_opt`. So I do **not** write the generator term, the discriminator optimization, the warm-up gate, or the weight balancing inside `VAELoss` — the loop owns all of that. My module's job is narrower: build the discriminator, set `disc_start`, keep my reconstruction terms, hand the loop the one tensor it needs to balance the GAN gradient, and add the one adversarial-family term the loop does *not* provide.

Three pieces, worked against the loop's actual code. First, the weight balancing. A constant weight on the adversarial term is guaranteed wrong somewhere in training: early the discriminator is near-random and its gradient into the decoder is tiny and noisy, later it sharpens and can swamp the reconstruction signal, and a fixed coefficient cannot track a moving ratio. What I want is for the adversarial and reconstruction gradients to arrive at the decoder with comparable magnitude automatically. They meet at the decoder's last layer, `vae.decoder.conv_out.weight`, through which every gradient into the decoder must pass; the loop measures both gradients *there* and sets the adaptive weight to $\|\nabla_{\text{ref}}\| / (\|\nabla_{g}\| + \delta)$, clamped and detached. The subtlety is the reference: the loop reads `criterion._perceptual_loss` if my module stashed it and uses *that* — the perceptual term alone — as the reference, falling back to the whole loss otherwise. So the adaptive weight normalizes the GAN gradient to the magnitude of the *perceptual* gradient at the last layer. My module must therefore store `self._perceptual_loss = p_loss`; if I forget, the loop balances against the entire loss and the GAN pressure is calibrated to the wrong scale. This is one concrete place the task differs from the generic lineage, where the adaptive weight balances against the full reconstruction loss.

Second, the discriminator. The quality problem is *local* — residual softness in texture and high-frequency detail — so I want a discriminator that judges patches, not one that collapses the whole image to a single real/fake scalar. A fully-convolutional PatchGAN outputs a grid of scores, one per receptive field, concentrating the adversarial signal on local texture realism and giving a dense gradient. I build it as `NLayerDiscriminatorWithFeatures`: the standard PatchGAN stack — strided conv, LeakyReLU, deeper strided convs with normalization — with two task-specific choices. I wrap every conv in **spectral normalization** to keep the discriminator Lipschitz and the adversarial game stable, which matters because the loop trains $D$ aggressively every step after warm-up, and I register forward hooks on the LeakyReLU activations to expose the **intermediate features**, because of the third piece.

Third — the one term the loop does not compute, so I add it inside `VAELoss` — **feature matching**. Beyond fooling the discriminator's final score, I want the reconstruction's *internal* discriminator features to match the real image's, layer by layer. Feature matching is a well-known stabilizer for adversarial training: instead of chasing only the single real/fake logit, the generator is asked to reproduce the statistics the discriminator extracts at each layer, a denser, less mode-collapse-prone signal that ties the adversarial objective back to perceptual structure. So when the discriminator is on, I run both target and reconstruction through $D$ with `return_features=True`, take the L1 distance between corresponding real and fake feature maps — detaching the real features, so the term trains the generator toward the discriminator's representation, not the discriminator toward the generator — average over the layers, and add it with $w_{\text{fm}} = 1.0$.

Assembling the fill: the reconstruction skeleton is unchanged — `F.l1_loss` anchor, frozen `lpips.LPIPS(net='vgg')` at $w_p = 0.5$ (float-cast, eval, no grad), closed-form KL at $\beta = 10^{-6}$ — and I stash `self._perceptual_loss = p_loss` for the loop's adaptive weight. I build `self.disc = NLayerDiscriminatorWithFeatures()` with its own `Adam(lr=1e-4, betas=(0.5, 0.9))` and set `self.disc_start = 5000`: the warm-up matters because a randomly initialized decoder produces garbage, and turning adversarial pressure on immediately lets $D$ win trivially and emit destructive gradients, so I hold the GAN off until the autoencoder reconstructs sanely. In `forward` I gate the feature-matching term on `step >= disc_start`, so it comes on exactly when the loop's own adversarial terms do. What I am deliberately *not* writing, because the loop handles it: the hinge discriminator loss $(\,\mathrm{relu}(1 + D(\text{recon.detach})) + \mathrm{relu}(1 - D(\text{real}))\,).\text{mean}()$, the **R1 gradient penalty** on real inputs ($\text{gp\_weight} = 10$), the generator logit loss $-\mathbb{E}[D(\text{recon})]$, and the adaptive-weight computation. The R1 penalty in particular is a task-specific stabilizer the lineage's pure-hinge recipe does not carry; it is in the loop, not mine to add. I expect the small scale's 19.64 to fall hardest again — the patch discriminator keeps finding the soft textures the static ruler tolerated — with medium and large ticking down by less, and the gmean dropping meaningfully below 9.31. The residual to attack next would be something the discriminator and perceptual term *together* still miss: a structured, frequency-localized error rather than generic softness.

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
