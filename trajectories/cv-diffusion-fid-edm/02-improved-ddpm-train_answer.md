The VP floor landed where I aimed it — CIFAR-10 FID of $2.48$ conditional and $3.01$ unconditional under the fixed deterministic sampler — and that left the headroom sitting exactly where the recipe inherited its choices from the chain rather than fitting them. Two of the floor's four decisions were never fitted: the reverse-process variance, which I pinned to an untrained schedule constant, and the noise schedule itself, the linear $\beta$ that decides how the cumulative signal coefficient $\bar\alpha_t$ decays and therefore where the training $\sigma$ land. Both are training-design choices, both DDPM took on faith, and both can be attacked without touching the sampler or the $\boldsymbol{\epsilon}$ target. That is the step I propose here: Improved DDPM, replacing those two inherited constants with two fitted decisions — a cosine noise schedule and a learned, per-coordinate reverse variance trained by a hybrid loss.

Take the schedule first, since it sets the stage the variance lives on. DDPM's forward process uses $\beta_t$ rising linearly from $10^{-4}$ to $0.02$, and the trouble is what $\bar\alpha_t$ does near the end of the chain under that schedule: it collapses to near zero well before $t=T$. The last stretch of timesteps is essentially pure noise already — the signal is gone, those steps teach the network almost nothing — while the early-to-middle steps where the interesting denoising happens are comparatively crowded. The linear schedule destroys the signal too fast. What I want is a schedule that decays $\bar\alpha_t$ more gently, spends fewer steps in the already-dead high-noise regime, and keeps the informative middle band wider. A cosine-squared decay does exactly that: set
$$\bar\alpha_t=\frac{f(t)}{f(0)},\qquad f(t)=\cos^2\!\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big).$$
The cosine is flat at both ends and steep in the middle, so $\bar\alpha_t$ leaves $1$ slowly, falls through the middle, and approaches $0$ slowly — no wasted near-pure-noise tail. The small offset $s=0.008$ is not free: I fix it so that $\sqrt{\beta_0}$ sits just below the pixel quantization bin width $1/127.5$, so the very first forward step already adds at least a pixel's worth of noise and the model is never asked to resolve sub-quantization differences.

I fold this schedule into the preconditioner the same way the floor folded the linear one — the $\sigma$-to-timestep map and the noise levels the network sees are read off the cosine $\bar\alpha_t$ instead of the exponential VP $\sigma(t)$. Concretely I precompute the discrete noise levels $u_j$ implied by the cosine $\bar\alpha$ ratios, stepping down from $j=M$ to $j=1$ via
$$u_{j-1}=\sqrt{\frac{u_j^2+1}{\mathrm{clip}\!\big(\bar\alpha_{j-1}/\bar\alpha_j,\;C_1\big)}-1},$$
with a small floor $C_1=0.001$ on the ratio so the recursion stays finite at the dead high-noise end. Then $c_{\text{noise}}$ maps a continuous $\sigma$ to the nearest of these $u_j$ by index. The denoiser path is identical to the floor: the input renormalization stays $c_{\text{in}}=1/\sqrt{\sigma^2+1}$, the output stays $c_{\text{out}}=-\sigma$, and $c_{\text{skip}}=1$. I have not changed the target, only the schedule that determines where the $\sigma$ land. The constant $C_2=0.008$ inside $\bar\alpha_j=\sin^2\!\big(\frac{\pi}{2}\frac{j/M}{C_2+1}\big)$ is the same offset $s$ doing the same job in the discrete indexing.

Now the variance, which is the larger structural change and the reason I touch the network's output head at all. In the floor I set the reverse variance to a fixed schedule constant — either $\beta_t$, the forward-step variance, or $\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$, the variance of the true forward posterior. These two natural endpoints differ by a large, $t$-dependent factor, and yet both gave nearly identical samples in the floor — sample quality is simply not very sensitive to this choice. But likelihood *is* sensitive to it, and a variance pinned to either endpoint is leaving information on the table at the steps where the right answer is somewhere between them. So rather than choose an endpoint, I let the network *interpolate* between them, per coordinate. The clean place for that interpolation is log space, because $\beta_t$ and $\tilde\beta_t$ span orders of magnitude: the network outputs a coefficient $v$ (one per pixel/channel) and I set
$$\Sigma_\theta=\exp\!\big(v\log\beta_t+(1-v)\log\tilde\beta_t\big).$$
With $v\in[0,1]$ this covers the whole bracket $[\tilde\beta_t,\beta_t]$, and the network learns, at each timestep and location, where in that band the variance should sit.

That means the raw network now emits *two* things per pixel: the noise prediction $\boldsymbol{\epsilon}_\theta$ as before and the variance coefficient $v$. So I double the output channels — the backbone produces $2C$ channels, the first $C$ feeding the denoiser exactly as in the floor, the second $C$ feeding $\Sigma_\theta$. The preconditioner's denoiser formula reads only the first half, $D=c_{\text{skip}}\mathbf{x}+c_{\text{out}}F_\theta[{:}C]$; the variance head is a side output used at sampling time, not in the denoiser. This is a strictly additive change to the head — the denoising path is untouched.

The catch is that the simple $\boldsymbol{\epsilon}$-MSE loss has *no gradient for* $v$. $L_{\text{simple}}$ is a function of the noise prediction only; the variance never appears in it, so doubling the head and training under $L_{\text{simple}}$ alone would leave $v$ flapping at its initialization. I need a loss term that depends on $\Sigma_\theta$, and the natural one is the variational bound $L_{\text{vlb}}$ — the per-timestep KL between the true posterior and the model's reverse step — which does depend on the variance. But training the full $L_{\text{vlb}}$ would put me back on the noisy, badly-weighted variational objective whose flat-MSE replacement was the whole reason the floor sampled well. So I cannot swap losses; I combine them. Train
$$L_{\text{hybrid}}=L_{\text{simple}}+\lambda L_{\text{vlb}},\qquad \lambda=0.001,$$
with $\lambda$ small enough that the dominant gradient on the mean stays the well-behaved flat $\boldsymbol{\epsilon}$-MSE and the variational term is present only enough to give $v$ a learning signal. And to keep $L_{\text{vlb}}$ from corrupting the mean, I apply a stop-gradient to $\boldsymbol{\mu}_\theta$ inside the $L_{\text{vlb}}$ term, so that term shapes $\Sigma_\theta$ and *only* $\Sigma_\theta$. The two heads are then cleanly separated: $L_{\text{simple}}$ trains the noise prediction, $\lambda L_{\text{vlb}}$ with the mean detached trains the variance.

The variational term creates one more problem, and it comes with its own fix. $L_{\text{vlb}}$ summed over timesteps is dominated by a few terms of enormous magnitude — its per-$t$ contributions are wildly uneven, so a uniform-$t$ Monte-Carlo estimate is high-variance, which makes the already-small $\lambda L_{\text{vlb}}$ gradient noisy and the variance head learn poorly. So I importance-sample its timesteps: draw $t$ with probability $p_t\propto\sqrt{\mathbb{E}[L_t^2]}$, estimating $\mathbb{E}[L_t^2]$ online from a running history of the last several loss values at each $t$, and divide by $p_t$ so the estimator stays unbiased. This concentrates samples on the high-magnitude timesteps and drives the variance of the $L_{\text{vlb}}$ estimate down. The importance sampling is *only* for the variational term — the mean's $L_{\text{simple}}$ keeps a uniform $t$ draw, because flat-over-$t$ is what made it sample well in the first place.

It is worth being honest about what this buys under the sampler I am scored on. The target is still $\boldsymbol{\epsilon}$ and the deterministic solver is fixed at NFE $=35$; that solver does not even consume the learned variance. So the FID gain here comes through the schedule — widening the informative band and not wasting steps on a dead tail — and through better optimization of the same denoiser, not through the variance changing the sampling rule. The cosine schedule should give a real FID improvement; the learned variance and hybrid loss buy likelihood and stabilize training but, under a variance-free deterministic sampler, contribute less to FID directly. The net I expect is a modest but genuine step below the floor in both settings, the improvement carried mostly by the schedule, with the variance/likelihood machinery laid in for the regime where it would matter more.

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
