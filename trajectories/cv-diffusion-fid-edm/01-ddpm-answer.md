**Problem.** Establish the denoiser-training floor on CIFAR-10 FID: with a fixed strong sampler
(NFE = 35) and a fixed U-Net backbone, pick the preconditioning, regression target, loss weighting,
and training-noise distribution that the standard variance-preserving (VP) diffusion recipe pins down.

**Key idea (DDPM, VP preconditioning + simple loss).** Work on the variance-preserving chain
$\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{y}+\sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}$ (linear
$\beta$ schedule, $\beta_1=10^{-4}\to\beta_T=0.02$, $T=1000$). Ask the network for the noise
$\boldsymbol{\epsilon}$ — the only quantity it does not already hold — and recover the denoiser via the
output scaling. Train the *unweighted* $\boldsymbol{\epsilon}$-MSE
$L_{\text{simple}}=\mathbb{E}_{t,\mathbf{y},\boldsymbol{\epsilon}}\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\|^2$
with $t$ drawn uniformly, which drops the variational weight that over-emphasizes the low-noise,
near-identity terms.

**Preconditioning.** $D=c_{\text{skip}}\mathbf{x}+c_{\text{out}}F_\theta(c_{\text{in}}\mathbf{x};c_{\text{noise}})$
with $c_{\text{skip}}=1$, $c_{\text{out}}=-\sigma$, $c_{\text{in}}=1/\sqrt{\sigma^2+1}$ (renormalize the
VP input to unit variance), and $c_{\text{noise}}=(M-1)\,\sigma^{-1}(\sigma)$ mapping $\sigma$ back to
the integer timestep. With $c_{\text{out}}=-\sigma$, requesting $D\approx\mathbf{y}$ is identical to
requesting $F_\theta\approx\boldsymbol{\epsilon}$.

**Loss weighting and noise distribution.** Draw $t$ uniformly on the schedule
($\sigma=\sigma(t)$, low end clamped at $\epsilon_t=10^{-5}$); weight the denoiser squared error by
$1/\sigma^2$, which makes $\frac{1}{\sigma^2}\|D-\mathbf{y}\|^2=\|F_\theta-\boldsymbol{\epsilon}\|^2$ —
the flat simple loss expressed through the denoiser interface. Reverse-process variance is a fixed
untrained schedule constant; no learned variance, schedule redesign, or augmentation.

**What to watch.** A solid but un-pushed FID in both the conditional and unconditional settings — the
baseline the later training-design changes must beat. Headroom is expected in the across-$\sigma$
emphasis (flat in $\boldsymbol{\epsilon}$-space is not uniform across noise levels) and in the
unfitted variance.

```python
# VP (variance-preserving) preconditioning: raw net predicts epsilon; D(x; sigma) is the denoiser.
class VPPrecond(torch.nn.Module):
    def __init__(self, img_resolution, img_channels, label_dim=0, use_fp16=False,
                 beta_d=19.9, beta_min=0.1, M=1000, epsilon_t=1e-5,
                 model_type='SongUNet', **model_kwargs):
        super().__init__()
        self.beta_d, self.beta_min, self.M, self.epsilon_t = beta_d, beta_min, M, epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels,
                                           out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)            # sigma -> integer timestep
        F_x = self.model((c_in * x), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        return c_skip * x + c_out * F_x                            # D(x; sigma)

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d


# Simple loss: uniform-t draw, 1/sigma^2 weight => flat epsilon-MSE through the denoiser interface.
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d, self.beta_min, self.epsilon_t = beta_d, beta_min, epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))   # uniform in t
        weight = 1 / sigma ** 2                                       # => epsilon-MSE
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        return weight * ((D_yn - y) ** 2)

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
```
