The task fixes everything except how the denoiser is trained: the U-Net backbone is given, the sampler is held at a strong deterministic NFE = 35, and CIFAR-10 FID over 50k samples is the only number that moves. What remains is a tangle of four coupled training-time decisions — how the raw network output is mapped to the denoiser as a function of $\sigma$, what regression target the loss compares against, how the per-$\sigma$ terms are weighted, and which $\sigma$ are drawn during training — each of which the field had made *inside* a particular forward-process chain rather than as a free knob. Before reaching for anything new, I have to pin down the recipe the whole field is built on, because that is the floor every later move has to beat: not invent the four choices freshly, but inherit them, intact, from the variance-preserving chain.

I propose **DDPM** — variance-preserving preconditioning paired with the simple $\boldsymbol{\epsilon}$-MSE loss — as that floor. Work on the variance-preserving marginal, where the signal is rescaled at every step so the noisy image keeps unit variance:
$$\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{y}+\sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon},$$
with $\bar\alpha_t$ the cumulative product of $(1-\beta_t)$ along a linear $\beta$ schedule from $10^{-4}$ to $0.02$ over $T=1000$ steps. This is the same family as the variance-exploding form $\mathbf{x}=\mathbf{y}+\sigma\boldsymbol{\epsilon}$, only written with the signal renormalized; the two are linked by $\sigma(t)=\sqrt{e^{\frac12\beta_d t^2+\beta_{\min}t}-1}$, with $\beta_d=19.9$, $\beta_{\min}=0.1$ encoding that linear schedule. I let the preconditioner carry this $t\leftrightarrow\sigma$ conversion so the rest of the machinery can speak in $\sigma$, and the point of starting VP is that all four decisions then come pre-made and can simply be read off.

The regression target has the most freedom, and it is where DDPM is most decisive. The ideal denoiser is $\mathbb{E}[\mathbf{y}\mid\mathbf{x}_t]$, and the reverse step regresses onto the forward-posterior mean. Substituting $\mathbf{y}=(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon})/\sqrt{\bar\alpha_t}$ collapses that mean to
$$\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol{\epsilon}\Big),$$
a function whose only unknown is $\boldsymbol{\epsilon}$ — the network already holds $\mathbf{x}_t$ as its own input. So the most economical thing to ask for is the one quantity it does not already have: the noise $\boldsymbol{\epsilon}$. Predict $\boldsymbol{\epsilon}$, and recovering everything else is algebra on $\mathbf{x}_t$. The reason to prefer this over asking for $\mathbf{y}$ directly is conditioning: at high noise $\mathbf{x}_t$ is almost all noise, so predicting $\boldsymbol{\epsilon}$ there is nearly the identity — the best-conditioned request exactly where $\mathbf{y}$ is least identifiable.

The weight is where the simplification that *defines* this recipe enters. Plugging the $\boldsymbol{\epsilon}$ parameterization into the variational reverse-process loss cancels the $\mathbf{x}_t$ terms and leaves a clean per-timestep $\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\|^2$ multiplied by a $t$-dependent variational coefficient $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$. I could train under that exact weight, but stare at what it does: it blows up at small $t$, near-zero noise, exactly where denoising is a near-identity that teaches the network almost nothing about image structure, and it under-weights the high-noise terms that set global layout. So I drop it and train the *unweighted* objective
$$L_{\text{simple}}=\mathbb{E}_{t,\mathbf{y},\boldsymbol{\epsilon}}\big[\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\|^2\big],$$
with $t$ drawn uniformly. This is not principled in a deep sense; it is the observation that a flat $\boldsymbol{\epsilon}$-MSE samples better than the weight the bound hands you, because flattening removes the low-noise over-emphasis.

What makes this fit the contract — and what takes the most care — is that the loss is given a *denoiser* output $D$, not a raw $\boldsymbol{\epsilon}$-prediction, so the preconditioner and the weight must interact precisely to reproduce the flat $\boldsymbol{\epsilon}$-MSE. I form the denoiser as $D=c_{\text{skip}}\mathbf{x}+c_{\text{out}}F_\theta(c_{\text{in}}\mathbf{x};c_{\text{noise}})$ with $c_{\text{skip}}=1$ and $c_{\text{out}}=-\sigma$. Then requesting $D\approx\mathbf{y}$ is identical to requesting $F_\theta\approx(\mathbf{x}-\mathbf{y})/\sigma=\boldsymbol{\epsilon}$ — the raw net predicts the noise, and the denoiser comes back for free. The input scaling must undo the VP signal rescaling so the network always sees unit variance, hence $c_{\text{in}}=1/\sqrt{\sigma^2+1}$. The conditioning scalar should be the discrete timestep the chain was trained against, so $c_{\text{noise}}=(M-1)\,\sigma^{-1}(\sigma)$ with $M=1000$ maps $\sigma$ back to the integer index. Finally, to make $D\approx\mathbf{y}$ equivalent to the flat $\boldsymbol{\epsilon}$-MSE, the loss weight on $\|D-\mathbf{y}\|^2$ must be $1/\sigma^2$: since $D-\mathbf{y}=c_{\text{out}}(F_\theta-\boldsymbol{\epsilon})=-\sigma(F_\theta-\boldsymbol{\epsilon})$, weighting by $1/\sigma^2$ gives back exactly $\|F_\theta-\boldsymbol{\epsilon}\|^2$. That is the simple loss, expressed through the denoiser interface.

The fourth decision, the $\sigma$-distribution, is forced once I commit to VP: draw $t$ uniformly on the schedule's support and set $\sigma=\sigma(t)$. A uniform-in-$t$ draw is *not* uniform in $\log\sigma$ — it concentrates where the schedule spends its steps — but it is the distribution the discrete chain was tuned for, so I take it as given rather than as a knob, clamping the low end to $\epsilon_t=10^{-5}$ so $\sigma$ stays bounded away from zero and the $1/\sigma^2$ weight stays finite. The reverse-process variance I do not fit at all: I set it to an untrained constant per timestep ($\beta_t$ or $\tilde\beta_t$ behave alike), because the object here is to establish the denoiser-training floor, and a learned variance is separate machinery that belongs to a later move.

Sanity-checking the across-$\sigma$ behavior this leaves: the input is always renormalized to unit variance by $c_{\text{in}}$; the target is $\boldsymbol{\epsilon}$ under a flat weight, so at high noise the request is well-conditioned and at low noise the recovery $\hat{\mathbf{y}}=(\mathbf{x}-\sqrt{1-\bar\alpha}\,\hat{\boldsymbol{\epsilon}})/\sqrt{\bar\alpha}$ divides a small noise error by a small $\sqrt{1-\bar\alpha}$, containing it. That sensible allocation is why the recipe works at all. But the flat weight is flat in $\boldsymbol{\epsilon}$-space, which after the $-\sigma$ output scaling is *not* a uniform emphasis across noise levels in denoiser-space, the $\sigma$-distribution is the chain's rather than one matched to where FID is sensitive, and the reverse variance is a guess. So I expect a solid but un-pushed FID in both the conditional and unconditional settings — the baseline against which every training-design change after this is measured, with headroom left precisely in the across-$\sigma$ emphasis and the unfitted variance.

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
