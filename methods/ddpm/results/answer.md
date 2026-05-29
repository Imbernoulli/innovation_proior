# Denoising Diffusion Probabilistic Models (DDPM)

## Problem

Generate high-quality images from a **likelihood-based, non-adversarial** model that is simple to define and stable to train. GANs sample well but are adversarial and give no likelihood; VAEs are blurry; flows are constrained; autoregressive models sample slowly; score-based models tune their sampler by hand. DDPM is a latent-variable model whose inference path is fixed (parameter-free) and whose reverse path is trained from variational terms.

## Key idea

Fix a forward Markov chain that gradually adds small Gaussian noise to the data over T steps until it becomes standard normal, and **learn the reverse chain** that denoises it. Because each forward step adds a tiny amount of noise, each reverse step is well-approximated by a Gaussian. Training reduces to teaching a single shared network to **predict the noise** added at a randomly chosen step — which is simultaneously (i) optimizing a variational bound, (ii) multi-scale denoising score matching, and (iii) training a Langevin-style sampler whose coefficients are pinned by the noise schedule.

## The method, cleanly

**Forward (fixed) process.** With β_1<…<β_T (linear, 1e-4→0.02), α_t=1−β_t, ᾱ_t=∏_{s≤t}α_s:

- single step: q(x_t|x_{t-1}) = N(√(1−β_t) x_{t-1}, β_t I) — the √(1−β_t) scaling keeps the marginal variance bounded.
- closed form (by induction): q(x_t|x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I), i.e. **x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε**, ε~N(0,I). Lets you sample any step directly.

**Reverse (learned) process.** p(x_T)=N(0,I), p_θ(x_{t-1}|x_t)=N(μ_θ(x_t,t), σ_t² I), σ_t² a fixed schedule constant (β_t or β̃_t; variances not learned).

**Variational bound**, after conditioning the forward posterior on x_0 and telescoping:

L = E_q[ KL(q(x_T|x_0)‖p(x_T)) + Σ_{t>1} KL(q(x_{t-1}|x_t,x_0)‖p_θ(x_{t-1}|x_t)) − log p_θ(x_0|x_1) ].

The first term is constant (q has no parameters). The forward posterior is the tractable Gaussian
q(x_{t-1}|x_t,x_0) = N(μ̃_t, β̃_t I), with β̃_t = (1−ᾱ_{t-1})/(1−ᾱ_t)·β_t and
μ̃_t = (√ᾱ_{t-1}β_t/(1−ᾱ_t)) x_0 + (√α_t(1−ᾱ_{t-1})/(1−ᾱ_t)) x_t,
so every middle term is a closed-form Gaussian KL: L_{t-1} = E[‖μ̃_t−μ_θ‖²/(2σ_t²)] + C.

**ε-parameterization.** Substituting x_0=(x_t−√(1−ᾱ_t)ε)/√ᾱ_t gives μ̃_t = (1/√α_t)(x_t − (β_t/√(1−ᾱ_t))ε). Since x_t is known, let the network predict ε:
μ_θ(x_t,t) = (1/√α_t)(x_t − (β_t/√(1−ᾱ_t)) ε_θ(x_t,t)).
Then L_{t-1}−C = E[ β_t²/(2σ_t²α_t(1−ᾱ_t)) · ‖ε − ε_θ(√ᾱ_t x_0+√(1−ᾱ_t)ε, t)‖² ], which is denoising score matching over noise scales t; the reverse step is a Langevin update with ε_θ as a learned score.

**Simplified objective (used for training):**

L_simple(θ) = E_{t,x_0,ε} ‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)‖²,  t ~ Uniform{1,…,T}.

Dropping the time-dependent weight removes the relative over-emphasis on the easy small-t (low-noise) terms and focuses capacity on harder high-noise denoising. Exact likelihood accounting still uses the discretized decoder term.

**Algorithms.**

- Training: sample x_0, t~Unif{1..T}, ε~N(0,I); gradient step on ‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t)ε, t)‖².
- Sampling: x_T~N(0,I); for t=T..1: z~N(0,I) if t>1 else 0; x_{t-1}=(1/√α_t)(x_t − ((1−α_t)/√(1−ᾱ_t)) ε_θ(x_t,t)) + σ_t z; return x_0.

**Architecture.** A single shared U-Net (PixelCNN++/Wide-ResNet backbone) handling all t; integer t encoded by Transformer sinusoidal embedding → MLP → injected into every residual block; self-attention at 16×16; group normalization. T=1000; data scaled to [−1,1]; adjacent 8-bit values are spaced 2/255 apart after scaling, so the last reverse step uses a discretized Gaussian decoder with half-width 1/255 for a proper discrete codelength. Adam (lr 2e-4), EMA decay 0.9999, dropout 0.1 (CIFAR), horizontal flips.

## Working code

```python
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn

def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

def extract(a, t, x_shape):
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, timesteps=1000, model_var_type="fixedlarge"):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.num_timesteps = timesteps
        self.channels = model.channels

        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        reg = lambda n, v: self.register_buffer(n, v.float())

        reg("betas", betas)
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1. / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance = torch.log(posterior_variance.clamp(min=1e-20))
        fixedlarge_variance = torch.cat([posterior_variance[1:2], betas[1:]])
        if model_var_type == "fixedsmall":
            model_log_variance = posterior_log_variance
        elif model_var_type == "fixedlarge":
            model_log_variance = torch.log(fixedlarge_variance.clamp(min=1e-20))
        else:
            raise ValueError("model_var_type must be 'fixedsmall' or 'fixedlarge'")

        reg("model_log_variance", model_log_variance)
        reg("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        reg("posterior_mean_coef2", (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_sample(self, x0, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def q_posterior_mean(self, x0, x_t, t):
        return (extract(self.posterior_mean_coef1, t, x_t.shape) * x0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        return F.mse_loss(self.model(x_t, t), noise)

    def forward(self, x0):
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device).long()
        return self.p_losses(x0, t)

    @torch.no_grad()
    def p_sample(self, x_t, t_int):
        t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
        eps = self.model(x_t, t)
        x0 = self.predict_start_from_noise(x_t, t, eps).clamp(-1., 1.)
        mean = self.q_posterior_mean(x0, x_t, t)
        log_var = extract(self.model_log_variance, t, x_t.shape)
        noise = torch.randn_like(x_t) if t_int > 0 else 0.
        return mean + (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def sample(self, batch_size=16):
        img = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=self.betas.device)
        for t_int in reversed(range(self.num_timesteps)):
            img = self.p_sample(img, t_int)
        return img

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

def group_norm(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))

class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim_out * 2))
        self.norm1 = group_norm(dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm2 = group_norm(dim_out)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        self.act = nn.SiLU()
        self.res = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.mlp(t_emb)[:, :, None, None].chunk(2, dim=1)
        h = h * (scale + 1) + shift
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.res(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, max_heads=4):
        super().__init__()
        heads = min(max_heads, channels)
        while channels % heads:
            heads -= 1
        self.heads = heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        scale = (c // self.heads) ** -0.5
        attn = torch.softmax(torch.einsum("bhdn,bhdm->bhnm", q * scale, k), dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)

class Unet(nn.Module):
    def __init__(self, dim=64, channels=3, dim_mults=(1, 2, 4)):
        super().__init__()
        self.channels = channels
        time_dim = dim * 4
        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim),
                                      nn.Linear(dim, time_dim), nn.GELU(),
                                      nn.Linear(time_dim, time_dim))
        dims = [dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList()
        skip_dims = []
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_dim),
                ResnetBlock(dim_out, dim_out, time_dim),
                Downsample(dim_out) if not is_last else nn.Identity(),
            ]))
            skip_dims.append(dim_out)

        mid_dim = dims[-1]
        self.mid1 = ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = AttentionBlock(mid_dim)
        self.mid2 = ResnetBlock(mid_dim, mid_dim, time_dim)

        self.ups = nn.ModuleList()
        cur_dim = mid_dim
        for i, skip_dim in enumerate(reversed(skip_dims)):
            is_last = i == len(skip_dims) - 1
            self.ups.append(nn.ModuleList([
                ResnetBlock(cur_dim + skip_dim, skip_dim, time_dim),
                ResnetBlock(skip_dim, skip_dim, time_dim),
                AttentionBlock(skip_dim) if i == 1 else nn.Identity(),
                Upsample(skip_dim) if not is_last else nn.Identity(),
            ]))
            cur_dim = skip_dim

        self.final_block = ResnetBlock(cur_dim, dim, time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        skips = []
        for block1, block2, down in self.downs:
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            skips.append(x)
            x = down(x)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        for block1, block2, attn, up in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            x = attn(x)
            x = up(x)
        return self.final_conv(self.final_block(x, t_emb))

class EMA:
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

def training_step(diffusion, x0, opt, ema=None):
    loss = diffusion(x0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if ema is not None:
        ema.update(diffusion)
    return loss.detach()

if __name__ == "__main__":
    model = Unet(dim=16, channels=3, dim_mults=(1, 2, 4))
    diffusion = GaussianDiffusion(model, image_size=32, timesteps=8)
    opt = torch.optim.Adam(diffusion.parameters(), lr=2e-4)
    ema = EMA(diffusion)
    x0 = torch.rand(2, 3, 32, 32) * 2 - 1
    loss = training_step(diffusion, x0, opt, ema)
    samples = ema.ema_model.sample(batch_size=2)
    print(float(loss), samples.shape)
```

The code uses the fixed-large reverse variance used in the sample-quality path; switching `model_var_type` to `"fixedsmall"` uses β̃_t instead. Exact bits/dim additionally evaluates the discretized-Gaussian L_0 term rather than replacing it with the simplified MSE surrogate.
