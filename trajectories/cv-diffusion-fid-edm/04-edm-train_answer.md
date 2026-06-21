The backbone has just been carried in strong enough that it can stay fixed: the ADM U-Net reaches FID 2.07 on ImageNet-64 on architecture alone, state of the art at that resolution, so the network is no longer what limits us. What is left is the part nobody had laid bare — the *training* of the denoiser is still the accreted chain recipe, three rungs of local patches (flat $\boldsymbol{\epsilon}$-MSE, a cosine schedule, learned variance) bolted onto VP/VE bookkeeping. The one limitation that forces the next idea is structural rather than numerical: the decisions that actually govern training — how the input and output are scaled in $\sigma$, what the network regresses toward, how much each noise level weighs in the loss, and which noise levels are sampled — are not free parameters in the existing recipes at all. They are tangled together inside the chain, each a side-effect of VP/VE algebra rather than a deliberate choice. So I stop patching and untangle them: name the four slots, derive each from first principles, and let them act together on the fixed network.

I propose **EDM** — a training design parameterized directly by $\sigma$ and $\sigma_{\text{data}}$, with no integer timesteps, no $\sigma$-to-$t$ inversion, and no schedule. It has four parts: a *preconditioning* of the network, a *loss weighting*, a *noise distribution*, and a *non-leaky augmentation*. Each falls out of a demand on the network's learning problem.

Before any of the principled design I want an honest starting line under *this* sampler, because the previous floor of 3.17 was measured with the old ancestral chain while I am scored under the fixed deterministic solver at NFE = 35. Retraining the existing VP/VE models cleanly and measuring them under the deterministic sampler is **config A**: CIFAR-10 conditional VP sits at FID 2.48, unconditional at 3.01. Then two non-conceptual moves clear out cruft that was never the point. **Config B** folds in hygiene the original runs got wrong — larger batches, a learning-rate ramp, a standardized EMA half-life, per-dataset dropout (13% on CIFAR), gradient clipping off — taking conditional VP to 2.18 and unconditional to 2.51; a real chunk of the headroom was just untuned knobs. **Config C** moves capacity: the lowest-resolution $4\times4$ layers do little for a $32\times32$ image, so I drop them and double the $16\times16$ layers that carry the semantically important mixing, holding the parameter count near 56M, reaching 2.08 / 2.31. From here every gain is genuinely about design.

Now the preconditioning, derived from scratch. I wrap the network as
$$D(\mathbf{x};\sigma)=c_{\text{skip}}(\sigma)\,\mathbf{x}+c_{\text{out}}(\sigma)\,F_\theta\big(c_{\text{in}}(\sigma)\,\mathbf{x};\,c_{\text{noise}}(\sigma)\big),$$
where the noisy input is $\mathbf{x}=\mathbf{y}+\sigma\boldsymbol{\epsilon}$ with $\mathbf{y}$ at data scale $\sigma_{\text{data}}$ and $\boldsymbol{\epsilon}$ unit-variance. The four scalars are exactly the input/output-scaling decision the chain keeps making for me — it hands over $c_{\text{out}}=-\sigma$, $c_{\text{in}}=1/\sqrt{\sigma^2+1}$, $c_{\text{skip}}=1$ — but those are consequences of VP bookkeeping, not requirements on the *network's* job. So I instead demand properties of that job and solve for the scalars.

The first demand is that the network's input have unit variance at every $\sigma$, so the net never sees inputs whose scale swings by orders of magnitude. The input is $c_{\text{in}}(\mathbf{y}+\sigma\boldsymbol{\epsilon})$, with variance $c_{\text{in}}^2(\sigma_{\text{data}}^2+\sigma^2)$, so unit variance forces
$$c_{\text{in}}(\sigma)=\frac{1}{\sqrt{\sigma_{\text{data}}^2+\sigma^2}}.$$
What changed from the chain's $1/\sqrt{\sigma^2+1}$ is that the $1$ became $\sigma_{\text{data}}^2$, the *actual* data variance. The chain assumed unit-variance data; real CIFAR pixels in $[-1,1]$ have variance $\approx0.25$, so $\sigma_{\text{data}}=0.5$, and using the true value is what makes the input normalization correct rather than approximate.

The second demand is that the network's *effective target* also have unit variance, because the quantity the net actually regresses is $F_\theta=(D-c_{\text{skip}}\mathbf{x})/c_{\text{out}}$; if its scale swings with $\sigma$ the loss is implicitly reweighting noise levels through the back door — the very tangle I am cutting. Setting $D=\mathbf{y}$ (the ideal denoiser), the target is $((1-c_{\text{skip}})\mathbf{y}-c_{\text{skip}}\sigma\boldsymbol{\epsilon})/c_{\text{out}}$, with variance $\big[(1-c_{\text{skip}})^2\sigma_{\text{data}}^2+c_{\text{skip}}^2\sigma^2\big]/c_{\text{out}}^2$. That is one unit-variance constraint on two free scalars, so I spend the remaining freedom on a second purpose: make the network's job easiest by *minimizing the amplification of its errors*, i.e. minimize $|c_{\text{out}}|$ subject to the constraint, so a given network error maps to the smallest possible denoiser error. Minimizing $c_{\text{out}}^2=(1-c_{\text{skip}})^2\sigma_{\text{data}}^2+c_{\text{skip}}^2\sigma^2$ over $c_{\text{skip}}$ gives
$$c_{\text{skip}}(\sigma)=\frac{\sigma_{\text{data}}^2}{\sigma^2+\sigma_{\text{data}}^2},\qquad c_{\text{out}}(\sigma)=\frac{\sigma\,\sigma_{\text{data}}}{\sqrt{\sigma_{\text{data}}^2+\sigma^2}}.$$
What makes this work is the way it reads at the two ends. At small $\sigma$, $c_{\text{skip}}\to1$ and $c_{\text{out}}\to0$: the denoiser is mostly identity-on-input plus a tiny correction, so the network predicts only the small residual — exactly right, since at low noise the input already nearly *is* the answer, and asking the net to reproduce it from scratch would amplify its error by $1/\sigma$, which is precisely the bug in the chain's $c_{\text{skip}}=1$, $c_{\text{out}}=-\sigma$. At large $\sigma$, $c_{\text{skip}}\to0$ and $c_{\text{out}}\to\sigma_{\text{data}}$: the denoiser leans entirely on the network's output, scaled to data variance, because the input carries no signal. So the interface is strictly better-conditioned across the whole range and interpolates *automatically* between predicting-the-noise and predicting-the-image as the noise level demands, instead of committing to one corner. The one scalar without a clean variational derivation is $c_{\text{noise}}$, taken empirically as $c_{\text{noise}}=\tfrac14\ln\sigma$ — the log compresses $\sigma$'s enormous dynamic range into a tame conditioning input, and the $\tfrac14$ is just what trains well; $\sigma$ is now first-class, conditioned on directly.

Swapping this principled preconditioning in for the chain's is **config D**, and on CIFAR it is essentially flat: 2.08 → 2.09 conditional, 2.31 → 2.29 unconditional. That is not a failure to read away — it is the point. The preconditioning makes the input and target unit-variance and well-conditioned at every $\sigma$, which *disentangles* the scaling from the weighting so the next change is clean (and at $64\times64$, where the dynamic range is wider, the same preconditioning helps materially on its own, so it is not cosmetic). With the scaling fixed, the loss weighting and the $\sigma$-distribution are the last tangle. The loss is $\mathbb{E}_\sigma\big[w(\sigma)\,\mathbb{E}_{\mathbf{y},\boldsymbol{\epsilon}}\|D(\mathbf{y}+\sigma\boldsymbol{\epsilon};\sigma)-\mathbf{y}\|^2\big]$, and since $D-\mathbf{y}=c_{\text{out}}(F_\theta-\text{target})$, the loss actually seen by the network is $w(\sigma)\,c_{\text{out}}(\sigma)^2\,\|F_\theta-\text{target}\|^2$. The effective weight on the network's unit-variance regression is therefore $w(\sigma)\,c_{\text{out}}(\sigma)^2$. I want every $\sigma$ to contribute equally to the gradient at initialization — there is no a-priori reason a noise level deserves more emphasis, and the chain's weighting was an accident of VP algebra — so I demand $w(\sigma)\,c_{\text{out}}(\sigma)^2=1$:
$$w(\sigma)=\frac{1}{c_{\text{out}}(\sigma)^2}=\frac{\sigma^2+\sigma_{\text{data}}^2}{(\sigma\,\sigma_{\text{data}})^2}.$$
This exactly cancels the $c_{\text{out}}^2$ amplification, turning the target's unit variance into a uniform per-$\sigma$ loss by construction.

Uniform weight is only half of it; I also choose *which* $\sigma$ to sample, because the noise levels are not equally *learnable*. At the extremes — near-zero noise, near-pure noise — the task is trivial or hopeless and there is nothing to learn; the useful signal lives in an intermediate band. So I draw $\ln\sigma\sim\mathcal{N}(P_{\text{mean}},P_{\text{std}}^2)$, a log-normal that puts a smooth hump over that band and tapers off the dead ends, with $P_{\text{mean}}=-1.2$ (since $e^{-1.2}\approx0.30$ sits squarely in the informative range for $\sigma_{\text{data}}=0.5$) and $P_{\text{std}}=1.2$ (wide enough to cover the useful range without wasting draws). This replaces the schedule-derived noise distribution entirely — no cosine, no linear $\beta$, no integer timesteps. Uniform weighting and log-normal sampling together are **config E**, and this is where the principled design pays off: conditional VP 2.09 → 1.88, unconditional 2.29 → 2.05, the largest single move in the climb — the move config D's disentangling made possible. And there is a bonus: once the loss and noise distribution are chosen this way, VP and VE stop mattering as training recipes. Their whole distinction was about how the chain distributes noise and scales signal; with the preconditioning and $\sigma$-distribution chosen on their own terms, configs E for VP and VE differ only in the backbone $F_\theta$. There is one training recipe.

One slot remains: augmentation. CIFAR-10 is small and a denoiser this strong overfits, so geometric augmentation — flips, scaling, rotation, anisotropy, translation — is the obvious regularizer, but naive augmentation *backfires* for a generative model. Train normally on augmented data and the model learns the *augmented* distribution and generates rotated, flipped, scaled images at inference; the augmentation leaks into the samples and FID against the clean reference gets worse. The fix is to make it *non-leaky*: feed the augmentation parameters to the network as a conditioning input, so it learns the augmented distribution *conditioned on the transform*, then at inference set that conditioning to zero — the "no augmentation" point — so the model generates only un-augmented images while still having benefited from the augmented data. Concretely the pipeline applies six geometric transforms at a modest probability (12% on CIFAR), packs their parameters into a 9-dimensional vector fed in alongside the noise level, and zeroes that vector at sampling time (and since it includes flips, I disable the dataset-level horizontal flip so the only flips are the conditioned, non-leaky ones). This is **config F**, the finale: conditional VP 1.88 → 1.79, unconditional 2.05 → 1.97.

Walking the whole additive climb in one breath — A 2.48 (honest baseline under this sampler), B 2.18 (tuned hyperparameters), C 2.08 (capacity to $16\times16$), D 2.09 (principled preconditioning, the disentangling null move), E 1.88 (uniform $1/c_{\text{out}}^2$ weighting + log-normal $\sigma$-sampling), F 1.79 (non-leaky augmentation), with the unconditional column tracking it 3.01 → 1.97 — the bar this trajectory was climbing toward is **CIFAR-10 FID 1.79 class-conditional and 1.97 unconditional** at NFE = 35, reached not with a cleverer sampler and not with a bigger network but by untangling the four training-design decisions and deriving each: unit-variance input gives $c_{\text{in}}$, unit-variance target plus minimal error amplification gives the $(c_{\text{skip}},c_{\text{out}})$ pair, equal per-$\sigma$ contribution gives $w(\sigma)=1/c_{\text{out}}^2$, learnability gives the log-normal $\sigma$-draw, and overfitting-without-leakage gives the conditioned augmentation. Carried unchanged onto the ADM ImageNet-$64$ backbone, the same training-design changes take its FID from 2.07 to a new state of the art, **1.36**.

```python
# EDM preconditioning: c_in/c_out/c_skip/c_noise solved from unit-variance demands; sigma is first-class.
class EDMPrecond(torch.nn.Module):
    def __init__(self, img_resolution, img_channels, label_dim=0, use_fp16=False,
                 sigma_min=0, sigma_max=float('inf'), sigma_data=0.5,
                 model_type='DhariwalUNet', **model_kwargs):
        super().__init__()
        self.sigma_min, self.sigma_max, self.sigma_data = sigma_min, sigma_max, sigma_data
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels,
                                           out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x = self.model((c_in * x), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        return c_skip * x + c_out * F_x                               # D(x; sigma)


# EDM loss: log-normal sigma draw; weight w(sigma) = 1/c_out^2 = (sigma^2 + sigma_data^2)/(sigma*sigma_data)^2.
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean, self.P_std, self.sigma_data = P_mean, P_std, sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()        # ln sigma ~ N(P_mean, P_std^2)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        return weight * ((D_yn - y) ** 2)


# Non-leaky augmentation: geometric transforms whose parameters condition the net (9-dim, zero at
# inference). Wired with p=0.12 on CIFAR; augment_dim=9; dataset-level x-flip disabled.
#   augment_kwargs = dict(class_name='training.augment.AugmentPipe', p=0.12,
#                         xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
#   network_kwargs.augment_dim = 9
class AugmentPipe:
    def __call__(self, images):
        # apply x/y-flip, isotropic scale, fractional rotation, anisotropy, fractional translation
        # at probability self.p; collect their parameters into a conditioning label vector `labels`;
        # return (augmented_images, labels). At sampling time the model is queried with labels = 0.
        ...
        return images, augment_labels
```
