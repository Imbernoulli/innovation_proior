I wanted a generative model that throws off sharp, globally coherent images — the kind of $32\times32$ CIFAR-10 samples that until now only adversarial models could produce — but without paying the adversarial tax. The moment a discriminator outpaces its generator the min-max diverges, mode collapse becomes a tuning séance, and when training stops I am left holding no number that tells me how well the model actually fits the data. I want instead a single scalar loss that plain SGD pushes down monotonically, with no second player; I want it likelihood-based, so there is a real codelength in bits per dimension to look at; and I want the definition simple enough to scale without inventing a new bag of tricks each time. None of the existing roads delivers all of this. The autoregressive PixelCNN family has beautiful likelihoods but samples one pixel at a time over thousands of sequential network calls, and its raster-scan order is an inductive bias inserted out of nowhere. Normalizing flows give exact likelihood and fast sampling but pin down each layer's expressiveness through invertibility and a tractable Jacobian, and force the latent to match the data dimension exactly. VAEs are the cleanest to write — the ELBO falls out the instant you apply Jensen — but a single amortized Gaussian posterior keeps collapsing, the decoder learns to ignore the latent, and the samples come out blurry, because one latent layer is too short a bridge from a complicated image distribution to a simple prior. The noise-conditional score networks finally reach near-GAN sample quality, but their sampler step sizes and per-scale noise are dialed in by hand after training, there is no proper likelihood, and the training objective never directly optimizes the sampler that is actually run.

So the question sharpens to this: can I build a latent-variable model whose inference path carries no parameters at all — nothing to collapse — trained by an honest variational bound so that it is likelihood-based and non-adversarial, and whose samples still land in the GAN tier? The parameterized inference network is the root of the VAE's troubles, so I simply refuse to learn the inference side. I propose Denoising Diffusion Probabilistic Models (DDPM): fix a forward process that grinds the data down into noise over many steps, and learn only how to run that process backwards. The generative model is $p_\theta(x_0)=\int p_\theta(x_{0:T})\,dx_{1:T}$, with latents $x_1,\dots,x_T$ all the same shape as the image, and the reverse process is a Markov chain of learned Gaussian transitions started from a fixed standard-normal prior, $$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t),\qquad p(x_T)=\mathcal N(0,I),\qquad p_\theta(x_{t-1}\mid x_t)=\mathcal N\big(\mu_\theta(x_t,t),\,\Sigma_\theta(x_t,t)\big).$$ The forward process I do not learn. The naive noising step $q(x_t\mid x_{t-1})=\mathcal N(x_t;x_{t-1},\beta_t I)$ lets the variance creep up every step, so $x_t$ drifts to ever larger scale and the single shared network sees inputs whose magnitude depends on $t$ — exactly the failure of the score-matching road, which adds noise without rescaling. So while I add noise I shrink the surviving signal in proportion: $$q(x_t\mid x_{t-1}) = \mathcal N\big(x_t;\,\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\big).$$ With $\mathrm{Var}(x_{t-1})=1$ this gives $\mathrm{Var}(x_t)=(1-\beta_t)\cdot 1+\beta_t=1$: the variance holds step after step, and that $\sqrt{1-\beta_t}$ factor is precisely what keeps the marginal near unit scale so the one network always sees consistent inputs.

I need to sample for training without walking $t$ forward steps when $t$ can be a thousand. Writing $\alpha_t:=1-\beta_t$ and $\bar\alpha_t:=\prod_{s=1}^t\alpha_s$, the per-step recursion $x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{\beta_t}\,\varepsilon$ unrolls — independent Gaussians add by adding variances, $\alpha_t(1-\alpha_{t-1})+(1-\alpha_t)=1-\alpha_t\alpha_{t-1}$ — to a closed-form marginal $$q(x_t\mid x_0)=\mathcal N\big(x_t;\,\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I\big),\qquad x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\ \ \varepsilon\sim\mathcal N(0,I).$$ This is the engine: I can jump straight to any $t$ in one shot, draw $t$ at random, and do SGD on a single random term of the bound. It is also the reparameterization trick used verbatim — the draw is a deterministic function of $x_0$ plus fixed scaled noise, so everything stays differentiable with low-variance gradients.

For the objective I bound the negative log-likelihood, $\mathbb E[-\log p_\theta(x_0)]\le \mathbb E_q[-\log(p_\theta(x_{0:T})/q(x_{1:T}\mid x_0))]=:L$. Optimizing $L$ as written fails: each term's denominator $q(x_t\mid x_{t-1})$ forgets $x_0$, so a Monte-Carlo estimate over the forward noise has brutal variance. But during training $x_0$ is right there, and because the forward chain is Markov, conditioning on it changes nothing — $q(x_t\mid x_{t-1})=q(x_t\mid x_{t-1},x_0)$ — so Bayes lets me flip it to $q(x_{t-1}\mid x_t,x_0)\,q(x_t\mid x_0)/q(x_{t-1}\mid x_0)$. Substituting and telescoping the $q(x_{t-1}\mid x_0)/q(x_t\mid x_0)$ ratios collapses the bound to $$L=\mathbb E_q\Big[\underbrace{\mathrm{KL}\big(q(x_T\mid x_0)\,\Vert\,p(x_T)\big)}_{L_T} + \sum_{t>1}\underbrace{\mathrm{KL}\big(q(x_{t-1}\mid x_t,x_0)\,\Vert\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}} - \underbrace{\log p_\theta(x_0\mid x_1)}_{-L_0}\Big].$$ Conditioning the forward posterior on $x_0$ is what Rao-Blackwellizes the whole thing: every middle term is now a KL between two Gaussians with a closed form, and the exploding variance is gone. Completing the square on $q(x_{t-1}\mid x_t,x_0)\propto q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)$ gives the explicit regression target $q(x_{t-1}\mid x_t,x_0)=\mathcal N(x_{t-1};\tilde\mu_t,\tilde\beta_t I)$ with $$\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t,\qquad \tilde\mu_t(x_t,x_0)=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t.$$ The term $L_T$ contains no $\theta$ — I fixed the forward $q$ and the $\beta_t$ are constants — so it is a discarded constant during training. That is a deliberate design choice: I could, like a VAE encoder, learn the $\beta_t$ by reparameterization, but fixing them makes $q$ wholly parameterless, so there is no such thing as posterior collapse, and it costs nothing because the forward chain is only a tool for grinding the data down.

For the reverse variance I take the cheapest route and set $\Sigma_\theta=\sigma_t^2 I$ to a non-learned schedule constant. The two natural candidates $\sigma_t^2=\beta_t$ and $\sigma_t^2=\tilde\beta_t$ bracket the situation — optimal respectively if $x_0$ were standard normal or a single deterministic point, the upper and lower entropy bounds — and I do not want to spend network capacity learning a per-pixel, per-timestep scale before the network has even learned which direction to denoise. With variance fixed, every variance and log-determinant piece of the Gaussian KL is $\theta$-independent, and $L_{t-1}=\mathbb E_q[(1/2\sigma_t^2)\Vert\tilde\mu_t-\mu_\theta\Vert^2]+C$. The most direct parameterization would have $\mu_\theta$ predict $\tilde\mu_t$, but $\tilde\mu_t$ is a linear blend of $x_0$ and $x_t$, and the network already has $x_t$ as input. Substituting the reparameterized $x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon)/\sqrt{\bar\alpha_t}$ and recombining shows $$\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon\Big),$$ so the target splits into a piece in $x_t$ the network gets for free and a piece in $\varepsilon$, the one thing it does not know. So I let the network predict $\varepsilon$ directly: $$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\Big),$$ and the $x_t$ terms cancel in the squared difference, leaving $$L_{t-1}-C=\mathbb E_{x_0,\varepsilon}\Big[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\big\Vert\varepsilon-\varepsilon_\theta(\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\,t)\big\Vert^2\Big].$$ This is denoising score matching summed over noise scales: since $\nabla_{x_t}\log q(x_t\mid x_0)=-\varepsilon/\sqrt{1-\bar\alpha_t}$, predicting $\varepsilon$ is predicting a scaled score, and the reverse draw $x_{t-1}=\mu_\theta+\sigma_t z$ is a Langevin update whose step sizes and noise are pinned by $\beta_t$ rather than hand-tuned. Three views — fitting the ELBO, doing denoising score matching, training a Langevin sampler — turn out to be one objective, and predicting $\varepsilon$ rather than $x_0$ keeps the target standardized across all $t$, which is why I prefer it over the equally valid $x_0$- and $x_{t-1}$-prediction branches.

The final term $L_0=-\log p_\theta(x_0\mid x_1)$ handles the discreteness of the data. The integer pixels $\{0,\dots,255\}$ are scaled to $[-1,1]$, where adjacent values sit $2/255$ apart so the bin around an interior value has half-width $1/255$. Laying my continuous final Gaussian over discrete data gives no proper codelength, so I integrate the density over each bin, $p_\theta(x_0\mid x_1)=\prod_i\int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)}\mathcal N(x;\mu_\theta^i(x_1,1),\sigma_1^2)\,dx$, with the end bins extending to $\pm\infty$; the result is a genuine lossless codelength with no dequantization noise and no scaling Jacobian smuggled in, and at the end of sampling I display $\mu_\theta(x_1,1)$ noiselessly. Finally, the per-$t$ weight $\beta_t^2/(2\sigma_t^2\alpha_t(1-\bar\alpha_t))$ leans hardest on the small-$t$, almost-no-noise terms — the easy near-identity denoising — steering capacity away from the high-noise steps where the network must actually invent structure. So I drop the weight to one: $$L_{\mathrm{simple}}(\theta)=\mathbb E_{t,x_0,\varepsilon}\big[\big\Vert\varepsilon-\varepsilon_\theta(\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\,t)\big\Vert^2\big],\qquad t\sim\mathrm{Uniform}\{1,\dots,T\}.$$ This removes the relative over-emphasis on the easy low-noise terms and redirects capacity toward the hard high-noise denoising that decides global structure; it is still a reweighted variational bound, and it coincides with the constant-weight denoising-score-matching objective of the score-based road.

Three implementation choices remain. I take $T=1000$ so the number of sampling passes matches the prior chain- and score-based work, with $\beta$ ramping linearly from $\beta_1=10^{-4}$ to $\beta_T=0.02$; small $\beta$ keeps the reverse conditional approximately Gaussian and drives $L_T=\mathrm{KL}(q(x_T\mid x_0)\Vert\mathcal N(0,I))$ down to about $10^{-5}$ bits per dimension, so the forward chain truly destroys the signal and the prior matches the aggregate posterior with no start-distribution drift. The network is a U-Net over Wide-ResNet blocks shaped like the unmasked PixelCNN++ backbone, whose skip connections let high-frequency detail bypass the bottleneck a denoiser must restore; one shared network handles all $t$, since $\bar\alpha_t$ makes the denoising task a smooth family in $t$. I tell it the timestep with a sinusoidal embedding pushed through a small MLP and added into every residual block, so nearby $t$ map to nearby codes and every layer can self-modulate to the current noise scale. I place self-attention only at the $16\times16$ resolution — coarse enough that the $O(n^2)$ cost is affordable, fine enough that there is real global layout to coordinate — and leave local texture to convolutions; concretely four feature resolutions $32\to16\to8\to4$ with attention at the second. I use group normalization, which is independent of batch size and noise scale, and dropout $0.1$ on CIFAR to curb the overfitting artifacts an unregularized backbone shows. Optimization is unremarkable: Adam at learning rate $2\times10^{-4}$, a parameter exponential moving average with decay $0.9999$ for the weights I sample from, batch size $128$, and random horizontal flips.

```python
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam


def get_step_variance_schedule(schedule, *, start, end, steps):
    if schedule == "linear":
        return torch.linspace(start, end, steps, dtype=torch.float64)
    if schedule == "quad":
        return torch.linspace(start ** 0.5, end ** 0.5, steps, dtype=torch.float64) ** 2
    if schedule == "const":
        return torch.full((steps,), end, dtype=torch.float64)
    raise NotImplementedError(schedule)


def extract(a, t, x_shape):
    return a.gather(0, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def mean_flat(x):
    return x.mean(dim=tuple(range(1, x.ndim)))


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) +
                  (mean1 - mean2).pow(2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    centered = x - means
    inv_std = torch.exp(-log_scales)
    cdf_plus = approx_standard_normal_cdf(inv_std * (centered + 1.0 / 255.0))
    cdf_min = approx_standard_normal_cdf(inv_std * (centered - 1.0 / 255.0))
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    return torch.where(
        x < -0.999, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )


class GaussianDiffusion(nn.Module):
    def __init__(self, backbone, image_size, num_timesteps=1000, schedule="linear",
                 beta_start=1e-4, beta_end=0.02, model_mean_type="eps",
                 model_var_type="fixedlarge", loss_type="mse"):
        super().__init__()
        assert model_mean_type in {"xprev", "xstart", "eps"}
        assert model_var_type in {"learned", "fixedsmall", "fixedlarge"}
        assert loss_type in {"kl", "mse"}
        self.backbone = backbone
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        self.channels = backbone.channels
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        betas = get_step_variance_schedule(
            schedule, start=beta_start, end=beta_end, steps=num_timesteps).float()
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        reg = lambda n, v: self.register_buffer(n, v.float())

        reg("betas", betas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        reg("log_one_minus_alphas_cumprod", torch.log(1. - alphas_cumprod))
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1. / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        reg("posterior_variance", posterior_variance)
        reg("posterior_log_variance_clipped",
            torch.log(torch.cat([posterior_variance[1:2], posterior_variance[1:]])))
        reg("fixedlarge_log_variance",
            torch.log(torch.cat([posterior_variance[1:2], betas[1:]])))
        reg("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        reg("posterior_mean_coef2",
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_sample(self, x0, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def q_posterior(self, x0, x_t, t):
        mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_start_from_xprev(self, x_t, t, xprev):
        return (extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
                extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t)

    def p_mean_variance(self, x_t, t, clip_denoised=True):
        out = self.backbone(x_t, t)
        if self.model_var_type == "learned":
            out, model_log_var = out.chunk(2, dim=1)
            model_var = torch.exp(model_log_var)
        elif self.model_var_type == "fixedsmall":
            model_var = extract(self.posterior_variance, t, x_t.shape).expand_as(x_t)
            model_log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape).expand_as(x_t)
        else:
            model_var = extract(self.betas, t, x_t.shape).expand_as(x_t)
            model_log_var = extract(self.fixedlarge_log_variance, t, x_t.shape).expand_as(x_t)

        clip = (lambda y: y.clamp(-1., 1.)) if clip_denoised else (lambda y: y)
        if self.model_mean_type == "xprev":
            x0 = clip(self.predict_start_from_xprev(x_t, t, out))
            mean = out
        elif self.model_mean_type == "xstart":
            x0 = clip(out)
            mean, _, _ = self.q_posterior(x0, x_t, t)
        else:
            x0 = clip(self.predict_start_from_noise(x_t, t, out))
            mean, _, _ = self.q_posterior(x0, x_t, t)
        return mean, model_var, model_log_var, x0

    @torch.no_grad()
    def p_sample(self, x_t, t_int, clip_denoised=True):
        t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
        mean, _, log_var, _ = self.p_mean_variance(x_t, t, clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(self, batch_size=16, device=None):
        device = device or self.betas.device
        img = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        for t_int in reversed(range(self.num_timesteps)):
            img = self.p_sample(img, t_int)
        return img

    def vb_terms_bpd(self, x0, x_t, t):
        true_mean, _, true_log_var = self.q_posterior(x0, x_t, t)
        mean, _, log_var, _ = self.p_mean_variance(x_t, t, clip_denoised=False)
        kl = mean_flat(normal_kl(true_mean, true_log_var, mean, log_var)) / math.log(2.)
        nll = -discretized_gaussian_log_likelihood(x0, means=mean, log_scales=0.5 * log_var)
        nll = mean_flat(nll) / math.log(2.)
        return torch.where(t == 0, nll, kl)

    def training_losses(self, x0, t=None, noise=None):
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device).long()
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        if self.loss_type == "kl":
            return self.vb_terms_bpd(x0, x_t, t)
        if self.model_var_type == "learned":
            raise ValueError("the simplified MSE objective uses a fixed-variance branch")
        target = {"xprev": self.q_posterior(x0, x_t, t)[0], "xstart": x0, "eps": noise}[self.model_mean_type]
        return mean_flat((target - self.backbone(x_t, t)).pow(2))

    def forward(self, x0):
        return self.training_losses(x0).mean()


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def group_norm(channels, max_groups=32):
    groups = min(max_groups, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def zero_module(m):
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim, dropout):
        super().__init__()
        self.norm1 = group_norm(dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, dim_out)
        self.norm2 = group_norm(dim_out)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(dim_out, dim_out, 3, padding=1))
        self.act = nn.SiLU()
        self.res = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.res(x)


class AttentionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = group_norm(c)
        self.q = nn.Conv2d(c, c, 1)
        self.k = nn.Conv2d(c, c, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.proj = zero_module(nn.Conv2d(c, c, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        hn = self.norm(x)
        q = self.q(hn).reshape(b, c, h * w)
        k = self.k(hn).reshape(b, c, h * w)
        v = self.v(hn).reshape(b, c, h * w)
        attn = torch.softmax(torch.einsum("bcn,bcm->bnm", q * (c ** -0.5), k), dim=-1)
        out = torch.einsum("bnm,bcm->bcn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """Time-conditioned U-Net, self-attention at 16x16 (the 'standard' DDPM CIFAR config)."""
    def __init__(self, image_size=32, channels=3, base_channels=128, out_channels=None,
                 channel_mult=(1, 2, 2, 2), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        time_dim = base_channels * 4
        self.init_conv = nn.Conv2d(channels, base_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(TimeEmbedding(base_channels),
                                      nn.Linear(base_channels, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))

        self.downs = nn.ModuleList()
        hs = [base_channels]
        cur = base_channels
        res = image_size
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(cur, out_ch, time_dim, dropout))
                cur = out_ch
                attns.append(AttentionBlock(cur) if res in attn_resolutions else nn.Identity())
                hs.append(cur)
            down = Downsample(cur) if level != len(channel_mult) - 1 else nn.Identity()
            if level != len(channel_mult) - 1:
                hs.append(cur)
                res //= 2
            self.downs.append(nn.ModuleList([blocks, attns, down]))

        self.mid1 = ResnetBlock(cur, cur, time_dim, dropout)
        self.mid_attn = AttentionBlock(cur)
        self.mid2 = ResnetBlock(cur, cur, time_dim, dropout)

        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip = hs.pop()
                blocks.append(ResnetBlock(cur + skip, out_ch, time_dim, dropout))
                cur = out_ch
                attns.append(AttentionBlock(cur) if res in attn_resolutions else nn.Identity())
            up = Upsample(cur) if level != 0 else nn.Identity()
            if level != 0:
                res *= 2
            self.ups.append(nn.ModuleList([blocks, attns, up]))

        self.final_norm = group_norm(cur)
        self.final_conv = zero_module(nn.Conv2d(cur, self.out_channels, 3, padding=1))

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.init_conv(x)
        skips = [h]
        for level, (blocks, attns, down) in enumerate(self.downs):
            for block, attn in zip(blocks, attns):
                h = attn(block(h, t_emb))
                skips.append(h)
            if level != len(self.downs) - 1:
                h = down(h)
                skips.append(h)
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)
        for level, (blocks, attns, up) in enumerate(self.ups):
            for block, attn in zip(blocks, attns):
                h = torch.cat((h, skips.pop()), dim=1)
                h = attn(block(h, t_emb))
            h = up(h)
        return self.final_conv(F.silu(self.final_norm(h)))


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        online = model.state_dict()
        for name, value in self.ema_model.state_dict().items():
            src = online[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                value.copy_(src)


if __name__ == "__main__":
    net = UNet(image_size=32, base_channels=16, channel_mult=(1, 2),
               num_res_blocks=1, attn_resolutions=(16,), dropout=0.1)
    diffusion = GaussianDiffusion(net, image_size=32, num_timesteps=8)
    opt = Adam(diffusion.parameters(), lr=2e-4)
    ema = ModelEMA(diffusion)
    x0 = torch.rand(2, 3, 32, 32) * 2 - 1
    loss = diffusion(x0)
    opt.zero_grad(); loss.backward(); opt.step(); ema.update(diffusion)
    print(float(loss.detach()), ema.ema_model.sample(batch_size=2).shape)
```

The same architecture in the `diffusers` library — the deployment form of the "standard" CIFAR-10 baseline (`google/ddpm-cifar10-32`) — is a `UNet2DModel` with four resolution levels and self-attention only at the second ($16\times16$) level:

```python
from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=32, in_channels=3, out_channels=3,
    block_out_channels=(128, 256, 256, 256),          # 4 levels: 32 -> 16 -> 8 -> 4
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),  # attn at 16x16 only
    layers_per_block=2, norm_num_groups=32, norm_eps=1e-6, act_fn="silu",
    time_embedding_type="positional", flip_sin_to_cos=False, freq_shift=1,
    downsample_padding=0,
)  # returns an object with .sample = predicted epsilon, shape [B, 3, 32, 32]
```
