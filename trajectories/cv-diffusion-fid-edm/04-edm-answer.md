**Problem.** With a near-best denoiser backbone fixed and the sampler fixed (NFE = 35), push CIFAR-10
FID by redesigning the *training* of the denoiser — untangling the four training-design decisions
(input/output scaling in $\sigma$, regression target, per-$\sigma$ loss weighting, $\sigma$-distribution)
that the VP/VE chains had fused together, plus augmentation.

**Key idea (EDM).** Derive each decision from a demand on the network's learning problem rather than
inheriting it from a chain.

- **Preconditioning** $D=c_{\text{skip}}\mathbf{x}+c_{\text{out}}F_\theta(c_{\text{in}}\mathbf{x};c_{\text{noise}})$,
  solved from two demands — unit-variance input and unit-variance effective target — plus minimal error
  amplification:
  $$c_{\text{in}}=\frac{1}{\sqrt{\sigma_{\text{data}}^2+\sigma^2}},\quad
    c_{\text{skip}}=\frac{\sigma_{\text{data}}^2}{\sigma^2+\sigma_{\text{data}}^2},\quad
    c_{\text{out}}=\frac{\sigma\,\sigma_{\text{data}}}{\sqrt{\sigma_{\text{data}}^2+\sigma^2}},\quad
    c_{\text{noise}}=\tfrac14\ln\sigma,$$
  with $\sigma_{\text{data}}=0.5$. This interpolates automatically between predict-the-noise (high
  $\sigma$) and predict-the-image (low $\sigma$).
- **Loss weighting** chosen so each $\sigma$ contributes equally to the network's gradient:
  $w(\sigma)=1/c_{\text{out}}(\sigma)^2=(\sigma^2+\sigma_{\text{data}}^2)/(\sigma\,\sigma_{\text{data}})^2$,
  cancelling the $c_{\text{out}}^2$ amplification.
- **Noise distribution** chosen for learnability: $\ln\sigma\sim\mathcal{N}(P_{\text{mean}},P_{\text{std}}^2)$,
  $P_{\text{mean}}=-1.2$, $P_{\text{std}}=1.2$, concentrating draws on the informative intermediate band.
- **Non-leaky augmentation:** apply geometric transforms, feed their parameters to the network as a
  9-dim conditioning input, and zero that input at inference so the augmentation does not leak into
  generated samples (probability 12% on CIFAR; dataset-level x-flip disabled).

Once the loss and noise distribution are set this way, VP and VE differ only in the backbone — there is
one training recipe parameterized directly by $\sigma$ and $\sigma_{\text{data}}$.

**Additive ablation (CIFAR-10 FID, NFE = 35).** Conditional VP: A 2.48 → B 2.18 (tuned hyperparameters)
→ C 2.08 (redistribute capacity) → D 2.09 (principled preconditioning) → E **1.88** (loss weighting +
noise distribution) → F **1.79** (non-leaky augmentation). Unconditional VP: A 3.01 → F **1.97**.
**Result: CIFAR-10 FID 1.79 class-conditional, 1.97 unconditional** — state of the art at NFE = 35.
The same training-design changes carried onto the ADM ImageNet-$64$ backbone with no architectural
changes take its FID from 2.07 to a new SOTA **1.36**.

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
