# Denoising Diffusion Probabilistic Models (DDPM), distilled

DDPM is a likelihood-based, non-adversarial image generator. A **fixed** forward Markov chain
slowly adds Gaussian noise to an image over T steps until it becomes standard normal noise; a
**learned** reverse Markov chain of Gaussian denoising steps undoes it, generating images by
running backward from noise. It is, structurally, a T-layer hierarchical VAE whose encoder is
fixed and parameter-free, trained by a variational bound that reduces — under an ε-prediction
parameterization and a simplified unweighted objective — to a plain mean-squared error between the
injected noise and a network's prediction of it. The "standard" CIFAR-10 configuration is a
time-conditioned U-Net with self-attention only at the 16×16 feature resolution.

## Problem it solves

Unconditional high-fidelity image generation that is, all at once: GAN-quality in samples,
trained by a single scalar non-adversarial objective, likelihood-based (a proper bits-per-dimension
bound), and simple to define with a parameter-free inference path that cannot collapse.

## Setup

- **Reverse (generative) process**: p_θ(x_0)=∫p_θ(x_{0:T})dx_{1:T}, with
  p_θ(x_{0:T})=p(x_T)∏_t p_θ(x_{t-1}|x_t), p(x_T)=N(0,I), p_θ(x_{t-1}|x_t)=N(μ_θ(x_t,t),Σ_θ(x_t,t)).
- **Forward (fixed) process**: q(x_t|x_{t-1})=N(√(1−β_t) x_{t-1}, β_t I). The √(1−β_t) scaling
  conserves the marginal variance (≈1), giving the shared network a consistent input scale. The
  β_t are fixed hyperparameters, not learned, so q has no parameters and L_T drops out of training.
- **Closed-form marginal** (by induction, α_t=1−β_t, ᾱ_t=∏_{s≤t}α_s):
  q(x_t|x_0)=N(√ᾱ_t x_0, (1−ᾱ_t)I), i.e. x_t=√ᾱ_t x_0+√(1−ᾱ_t)ε, ε~N(0,I). Lets training jump to
  any t in one step.

## Key idea: the variational bound, reduced

Bound the NLL and rewrite by conditioning the forward posterior on x_0 (telescoping the
q(x_{t-1}|x_0)/q(x_t|x_0) ratios):

```
L = E_q[ KL(q(x_T|x_0)‖p(x_T))               # L_T  : constant (beta fixed) -> dropped
       + sum_{t>1} KL(q(x_{t-1}|x_t,x_0)‖p_θ(x_{t-1}|x_t))   # L_{t-1}: Gaussian-vs-Gaussian KL
       - log p_θ(x_0|x_1) ]                   # L_0  : discrete decoder
```

Conditioning on x_0 makes every L_{t-1} a closed-form KL between two Gaussians (Rao-Blackwellized,
no high-variance Monte Carlo). The forward posterior, by completing the square, is
q(x_{t-1}|x_t,x_0)=N(μ̃_t, β̃_t I) with

```
beta_tilde_t = (1 - bar_a_{t-1})/(1 - bar_a_t) * beta_t
mu_tilde_t(x_t,x0) = (sqrt(bar_a_{t-1}) beta_t / (1 - bar_a_t)) x0
                   + (sqrt(a_t) (1 - bar_a_{t-1}) / (1 - bar_a_t)) x_t
```

**Fixed reverse variance.** Set Σ_θ=σ_t² I to an untrained constant; σ_t²=β_t and σ_t²=β̃_t are the
two natural extremes — optimal for x_0~N(0,I) and for x_0 a single point, the upper/lower entropy
bounds. The standard path fixes variance and learns only the mean, while the code keeps the learned
variance branch as an available Gaussian parameterization. Then
L_{t-1}=E_q[(1/2σ_t²)‖μ̃_t−μ_θ‖²]+C.

**ε-prediction.** Substitute x_0=(x_t−√(1−ᾱ_t)ε)/√ᾱ_t into μ̃_t:
μ̃_t=(1/√α_t)(x_t−(β_t/√(1−ᾱ_t))ε). Since x_t is the network's input, parameterize
μ_θ(x_t,t)=(1/√α_t)(x_t−(β_t/√(1−ᾱ_t))ε_θ(x_t,t)) — i.e. predict the noise. The term becomes

```
L_{t-1} - C = E_{x0,eps}[ beta_t^2 / (2 sigma_t^2 a_t (1 - bar_a_t))
                          * || eps - eps_θ(sqrt(bar_a_t) x0 + sqrt(1 - bar_a_t) eps, t) ||^2 ]
```

This is denoising score matching summed over noise scales (Vincent's identity:
∇_{x_t} log q(x_t|x_0) = −ε/√(1−ᾱ_t), so ε_θ is a learned scaled score), and the reverse step
x_{t-1}=μ_θ+σ_t z is Langevin dynamics with coefficients fixed by β_t rather than hand-tuned.

**Discrete decoder for L_0.** With 8-bit pixels scaled to [−1,1] (bins of half-width 1/255),
p_θ(x_0|x_1)=∏_i ∫_bin N(x; μ_θ^i(x_1,1), σ_1²) dx, giving a true lossless codelength without
dequantization noise; the end bins extend to ±∞.

**Simplified objective.** Drop the per-t weight to 1:

```
L_simple(θ) = E_{t,x0,eps}[ || eps - eps_θ(sqrt(bar_a_t) x0 + sqrt(1 - bar_a_t) eps, t) ||^2 ],
              t ~ Uniform{1, ..., T}
```

The true weight over-emphasizes the easy low-noise (small-t) terms; the unit weight redirects
capacity toward the harder high-noise denoising that controls global structure. It is still a
reweighted variational bound (β-VAE-style emphasis), and matches NCSN's denoising-score-matching
weighting.

## Defaults and why

- **T = 1000**, β linear from β_1=10^{-4} to β_T=0.02. Small β keeps the reverse step's functional
  form close to the forward's (so the Gaussian reverse is a valid local approximation) and keeps
  the SNR at x_T tiny, so L_T=KL(q(x_T|x_0)‖N(0,I))≈10^{-5} bits/dim≈0 — the forward truly destroys
  the signal and the prior matches the aggregate posterior. T need not equal the data dimension.
- **U-Net backbone** (PixelCNN++/Wide-ResNet shape) with skip connections, so high-frequency
  detail bypasses the bottleneck a denoiser must restore. **One shared network across all t**, since
  ᾱ_t makes the task a smooth family in t. **Group normalization** (batch- and noise-level
  independent). **Sinusoidal time embedding** added into **every** residual block (smooth
  multi-frequency code; per-layer modulation to the noise scale). **Self-attention at 16×16** only —
  O(n²) attention is affordable only at a coarse resolution, and 16×16 is where global layout can be
  coordinated while convolutions handle local texture. **Dropout 0.1** on CIFAR-10.
- **Optimizer**: Adam, lr 2×10^{-4} (2×10^{-5} at 256²), parameter EMA decay 0.9999, batch 128
  (64 at high resolution), random horizontal flips.

## Algorithms

```
Training:
  repeat:
    x0 ~ data;  t ~ Uniform{1..T};  eps ~ N(0, I)
    take gradient step on  || eps - eps_θ(sqrt(bar_a_t) x0 + sqrt(1 - bar_a_t) eps, t) ||^2

Sampling:
  x_T ~ N(0, I)
  for t = T, ..., 1:
    z ~ N(0, I) if t > 1 else 0
    x_{t-1} = (1/sqrt(a_t)) * ( x_t - ((1 - a_t)/sqrt(1 - bar_a_t)) * eps_θ(x_t, t) ) + sigma_t * z
  return x_0
```

## Connections

- **Autoregressive models** are a special case: in the equivalent bound
  L=KL(q(x_T)‖p(x_T))+Σ E_q KL(q(x_{t-1}|x_t)‖p_θ)+H(x_0), set T=dim, forward="mask the t-th
  coordinate," p(x_T)=blank, p_θ fully expressive → an AR model. So Gaussian diffusion is an AR
  model with a generalized, non-coordinate bit ordering, and the chain length is freed from the data
  dimension.
- **Progressive lossy coding**: L_1..L_T as rate, L_0 as distortion; the reverse process is a
  progressive decoder with running estimate x̂_0=(x_t−√(1−ᾱ_t)ε_θ(x_t))/√ᾱ_t.

## Working code

Grounded in the canonical implementation (identical buffers, coefficients, and the
fixedlarge/fixedsmall variance and eps/xstart/xprev prediction branches). The schedule-derived
coefficients are precomputed; `extract` gathers per-t coefficients and broadcasts to image shape.
The code is zero-indexed: array index `0` is the first one-based reverse step, where
`posterior_variance[0]` is zero, so the clipped posterior log variance and `fixedlarge_log_variance`
reuse `posterior_variance[1]` at index `0`; `fixedlarge_log_variance` then uses `betas[1:]` for the
remaining entries.

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

The same architecture in the `diffusers` library — the deployment form of the "standard" CIFAR-10
baseline (`google/ddpm-cifar10-32`) — is a `UNet2DModel` with four resolution levels and
self-attention only at the second (16×16) level:

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
