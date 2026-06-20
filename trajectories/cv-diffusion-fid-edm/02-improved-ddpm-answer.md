**Problem (from rung 1).** The VP floor fixed the across-$\sigma$ emphasis to the chain's defaults and
pinned two decisions to inherited constants: the reverse-process variance (an untrained schedule
scalar) and the noise schedule itself (linear $\beta$). Both waste capacity — the linear schedule
destroys the signal too fast and leaves a dead high-noise tail; the fixed variance is a guess. Fit
both.

**Key idea (Improved DDPM).** (1) **Cosine schedule:** replace the linear $\beta$ with
$\bar\alpha_t=f(t)/f(0)$, $f(t)=\cos^2\!\big(\tfrac{t/T+s}{1+s}\cdot\tfrac{\pi}{2}\big)$, $s=0.008$
(chosen so $\sqrt{\beta_0}$ sits just below the pixel bin width $1/127.5$), which decays $\bar\alpha_t$
gently at both ends and stops wasting the near-pure-noise tail. (2) **Learned reverse variance:** have
the network emit, per coordinate, a coefficient $v$ and set
$\Sigma_\theta=\exp\!\big(v\log\beta_t+(1-v)\log\tilde\beta_t\big)$, interpolating in log space between
the forward variance $\beta_t$ and the posterior variance $\tilde\beta_t$.

**Training.** The simple loss has no gradient for $v$, so train
$L_{\text{hybrid}}=L_{\text{simple}}+\lambda L_{\text{vlb}}$ with $\lambda=0.001$ and a stop-gradient on
$\boldsymbol{\mu}_\theta$ inside $L_{\text{vlb}}$ — the mean is learned by the flat
$\boldsymbol{\epsilon}$-MSE alone, the variational term shapes only $\Sigma_\theta$. The variational
term is high-variance across $t$, so **importance-sample** its timesteps with $p_t\propto\sqrt{\mathbb{E}[L_t^2]}$
(estimated from a running per-$t$ loss history) and divide by $p_t$; the mean's $L_{\text{simple}}$
keeps a uniform $t$ draw.

**Preconditioning.** Identical denoiser path to rung 1 — $c_{\text{skip}}=1$, $c_{\text{out}}=-\sigma$,
$c_{\text{in}}=1/\sqrt{\sigma^2+1}$ — but the backbone now outputs $2C$ channels: $D$ reads the first
$C$ (the $\boldsymbol{\epsilon}$-prediction); the second $C$ feed the variance head. $c_{\text{noise}}$
maps $\sigma$ to the nearest discrete level $u_j$ implied by the cosine $\bar\alpha$, precomputed by the
downward recursion with floors $C_1=0.001$, $C_2=0.008$.

**What to watch.** A CIFAR-10 FID a notch below the floor, carried mostly by the cosine schedule; the
learned variance and hybrid loss buy likelihood and stabilize training but contribute less to FID under
a variance-free deterministic sampler.

```python
# iDDPM preconditioning: cosine-schedule discrete noise levels u_j; net emits 2C channels,
# D reads the first C (epsilon-prediction). c_in/c_out/c_skip unchanged from VP.
class iDDPMPrecond(torch.nn.Module):
    def __init__(self, img_resolution, img_channels, label_dim=0, use_fp16=False,
                 C_1=0.001, C_2=0.008, M=1000, model_type='DhariwalUNet', **model_kwargs):
        super().__init__()
        self.C_1, self.C_2, self.M = C_1, C_2, M
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels,
                                           out_channels=img_channels * 2, label_dim=label_dim, **model_kwargs)
        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):                                   # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min, self.sigma_max = float(u[M - 1]), float(u[0])

    def forward(self, x, sigma, class_labels=None, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)
        F_x = self.model((c_in * x), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        return c_skip * x + c_out * F_x[:, :self.model.img_channels]  # denoiser reads first C channels

    def alpha_bar(self, j):                                          # cosine schedule
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        return (index if return_index else self.u[index.flatten()]).reshape(sigma.shape)
```
