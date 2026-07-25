The problem is to build an image generative model that is simultaneously high-quality, likelihood-based, non-adversarial, and simple to train. In mid-2020 no single family satisfied all of these. GANs produce sharp samples but rely on a min-max game, give no likelihood, and suffer from mode collapse and training instability. Autoregressive models optimize likelihood faithfully but sample one coordinate at a time, making generation expensive and imposing an arbitrary ordering. Normalizing flows offer exact likelihood and fast sampling, but tractable Jacobians and invertibility restrict expressiveness. VAEs have a clean variational objective and an amortized inference network, yet a single Gaussian posterior layer is too weak a bridge between complex data and a simple prior, leading to posterior collapse and blurry samples. Score-based methods such as NCSN train a noise-conditional score estimator and sample with annealed Langevin dynamics, but their step sizes, noise scales, and sampler coefficients are hand-tuned after training, the corrupted inputs are not rescaled so variances blow up, the chain does not end at a fixed known prior, and the approach lacks a variational bound or likelihood.

What is missing is a latent-variable model whose inference path is fixed and parameter-free, so there is nothing to collapse, while the generative path is trained by a genuine variational bound. Sohl-Dickstein et al. (2015) laid out the right skeleton: a fixed forward Markov chain that slowly adds noise until the data becomes pure noise, and a learned reverse chain that removes the noise step by step. Because each forward step is small, each reverse conditional is approximately Gaussian. But that work did not settle the parameterization of the reverse mean, the choice of reverse variance, the noise schedule, the architecture, or the objective weighting, and its image quality was weak. The new method fills in those choices.

The method is Denoising Diffusion Probabilistic Models, or DDPM. It fixes a forward process that corrupts data over T steps. A naive additive-noise step would make the variance grow, so instead use a variance-preserving step q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) x_{t-1}, beta_t I). Let alpha_t = 1 - beta_t and alpha_bar_t be the product of alphas up to t. Then by induction q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I), which means x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon for epsilon ~ N(0, I). This closed form lets training sample any timestep directly, reusing the reparameterization trick without walking the whole chain.

The generative model is the reverse Markov chain p_theta(x_{0:T}) = p(x_T) prod_t p_theta(x_{t-1} | x_t), with p(x_T) = N(0, I) and p_theta(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma_t^2 I). The reverse variance sigma_t^2 is held fixed rather than learned, because the mean carries the image structure. Writing the variational bound and conditioning the forward posterior on x_0 causes the terms to telescope, leaving a sum of closed-form Gaussian KLs plus a final decoder term. Completing the square gives the forward posterior q(x_{t-1} | x_t, x_0) = N(mu_tilde_t, beta_tilde_t I) with beta_tilde_t = (1 - alpha_bar_{t-1})/(1 - alpha_bar_t) beta_t and mu_tilde_t a linear combination of x_0 and x_t. Substituting the reparameterization of x_t in terms of epsilon reveals that the model only needs to predict the noise: mu_theta(x_t, t) = (1/sqrt(alpha_t)) (x_t - beta_t/sqrt(1 - alpha_bar_t) epsilon_theta(x_t, t)). Then the KL reduces to a weighted mean squared error between epsilon and epsilon_theta(x_t, t), which is exactly multi-scale denoising score matching derived from a variational bound rather than assembled by hand.

The full variational weight overemphasizes small-t, low-noise terms, which are nearly identity mappings and waste capacity. DDPM therefore trains on a simplified uniform objective: the expected squared error between the sampled noise and the network's prediction, with t drawn uniformly. For exact likelihood evaluation the discretized Gaussian decoder at the final step is still used, because pixels are 8-bit integers mapped to [-1, 1] and a continuous density must be integrated over bins of half-width 1/255. The reverse step has the shape of a Langevin update: move along the learned score epsilon_theta, then add a small amount of fresh noise, but the coefficients are pinned by the forward schedule rather than tuned post hoc.

The network is a single shared U-Net that handles all timesteps. Integer t is encoded with sinusoidal embeddings, passed through a small MLP, and added into every residual block so each layer adapts to the current noise scale. The backbone uses group normalization, dropout on CIFAR-10, skip connections to preserve detail, and self-attention at the 16x16 feature resolution to capture long-range structure without quadratic cost at fine scales. Training uses T = 1000, a linear beta schedule from 1e-4 to 0.02, data scaled to [-1, 1], Adam with learning rate 2e-4, and an exponential moving average of parameters with decay 0.9999. Sampling starts from standard normal noise and iteratively applies the reverse mean, adding scaled noise at every step except the last.

```python
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

def _warmup_schedule(start, end, steps, frac):
    betas = torch.full((steps,), end, dtype=torch.float64)
    warmup = int(steps * frac)
    betas[:warmup] = torch.linspace(start, end, warmup, dtype=torch.float64)
    return betas

def get_step_variance_schedule(schedule, *, start, end, steps):
    if schedule == "quad":
        return torch.linspace(start ** 0.5, end ** 0.5, steps, dtype=torch.float64) ** 2
    if schedule == "linear":
        return torch.linspace(start, end, steps, dtype=torch.float64)
    if schedule == "warmup10":
        return _warmup_schedule(start, end, steps, 0.1)
    if schedule == "warmup50":
        return _warmup_schedule(start, end, steps, 0.5)
    if schedule == "const":
        return torch.full((steps,), end, dtype=torch.float64)
    if schedule == "jsd":
        return 1.0 / torch.linspace(steps, 1, steps, dtype=torch.float64)
    raise NotImplementedError(schedule)

def extract(a, t, x_shape):
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

def mean_flat(x):
    return x.mean(dim=tuple(range(1, x.ndim)))

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 +
                  torch.exp(logvar1 - logvar2) +
                  (mean1 - mean2).pow(2) * torch.exp(-logvar2))

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    centered = x - means
    inv_std = torch.exp(-log_scales)
    plus_in = inv_std * (centered + 1.0 / 255.0)
    min_in = inv_std * (centered - 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    return torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )

class ImageLatentGenerator(nn.Module):
    def __init__(self, backbone, image_size, latent_steps=1000, schedule="linear",
                 variance_start=1e-4, variance_end=0.02, prediction_type="eps",
                 variance_type="fixedlarge", loss_type="mse"):
        super().__init__()
        if prediction_type not in {"xprev", "xstart", "eps"}:
            raise ValueError("prediction_type must be 'xprev', 'xstart', or 'eps'")
        if variance_type not in {"learned", "fixedsmall", "fixedlarge"}:
            raise ValueError("variance_type must be 'learned', 'fixedsmall', or 'fixedlarge'")
        if loss_type not in {"kl", "mse"}:
            raise ValueError("loss_type must be 'kl' or 'mse'")
        self.backbone = backbone
        self.image_size = image_size
        self.latent_steps = latent_steps
        self.channels = backbone.channels
        self.prediction_type = prediction_type
        self.variance_type = variance_type
        self.loss_type = loss_type

        betas = get_step_variance_schedule(
            schedule, start=variance_start, end=variance_end, steps=latent_steps).float()
        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError("all step variances must be in (0, 1]")
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
        posterior_log_variance_clipped = torch.log(torch.cat([posterior_variance[1:2],
                                                              posterior_variance[1:]]))
        fixedlarge_log_variance = torch.log(torch.cat([posterior_variance[1:2], betas[1:]]))
        reg("posterior_variance", posterior_variance)
        reg("posterior_log_variance_clipped", posterior_log_variance_clipped)
        reg("fixedlarge_log_variance", fixedlarge_log_variance)
        reg("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        reg("posterior_mean_coef2", (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_mean_variance(self, x0, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
        variance = extract(1. - self.alphas_cumprod, t, x0.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x0.shape)
        return mean, variance, log_variance

    def q_sample(self, x0, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def q_posterior_mean_variance(self, x0, x_t, t):
        mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        variance = extract(self.posterior_variance, t, x_t.shape)
        log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_start_from_xprev(self, x_t, t, xprev):
        return (extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
                extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t)

    def p_mean_variance(self, x_t, t, clip_denoised=True, return_pred_xstart=False):
        model_output = self.backbone(x_t, t)
        if self.variance_type == "learned":
            model_output, model_log_variance = model_output.chunk(2, dim=1)
            model_variance = torch.exp(model_log_variance)
        elif self.variance_type == "fixedsmall":
            model_variance = extract(self.posterior_variance, t, x_t.shape).expand_as(x_t)
            model_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape).expand_as(x_t)
        else:
            model_variance = extract(self.betas, t, x_t.shape).expand_as(x_t)
            model_log_variance = extract(self.fixedlarge_log_variance, t, x_t.shape).expand_as(x_t)

        maybe_clip = (lambda y: y.clamp(-1., 1.)) if clip_denoised else (lambda y: y)
        if self.prediction_type == "xprev":
            pred_xstart = maybe_clip(self.predict_start_from_xprev(x_t, t, model_output))
            model_mean = model_output
        elif self.prediction_type == "xstart":
            pred_xstart = maybe_clip(model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x_t, t)
        else:
            pred_xstart = maybe_clip(self.predict_start_from_noise(x_t, t, model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x_t, t)

        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, pred_xstart
        return model_mean, model_variance, model_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t_int, clip_denoised=True, return_pred_xstart=False):
        t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            x_t, t, clip_denoised=clip_denoised, return_pred_xstart=True)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return (sample, pred_xstart) if return_pred_xstart else sample

    @torch.no_grad()
    def sample(self, batch_size=16, device=None):
        device = device or self.betas.device
        img = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        for t_int in reversed(range(self.latent_steps)):
            img = self.p_sample(img, t_int)
        return img

    def vb_terms_bpd(self, x0, x_t, t, clip_denoised=True, return_pred_xstart=False):
        true_mean, _, true_log_variance = self.q_posterior_mean_variance(x0, x_t, t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            x_t, t, clip_denoised=clip_denoised, return_pred_xstart=True)
        kl = mean_flat(normal_kl(true_mean, true_log_variance, model_mean, model_log_variance)) / math.log(2.)
        decoder_nll = -discretized_gaussian_log_likelihood(
            x0, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = mean_flat(decoder_nll) / math.log(2.)
        out = torch.where(t == 0, decoder_nll, kl)
        return (out, pred_xstart) if return_pred_xstart else out

    def training_losses(self, x0, t=None, noise=None):
        if t is None:
            t = torch.randint(0, self.latent_steps, (x0.shape[0],), device=x0.device).long()
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        if self.loss_type == "kl":
            return self.vb_terms_bpd(x0, x_t, t, clip_denoised=False)
        if self.variance_type == "learned":
            raise ValueError("the simplified MSE loss uses a fixed variance branch")
        target = {
            "xprev": self.q_posterior_mean_variance(x0, x_t, t)[0],
            "xstart": x0,
            "eps": noise,
        }[self.prediction_type]
        model_output = self.backbone(x_t, t)
        return mean_flat((target - model_output).pow(2))

    def training_loss(self, x0, t=None, noise=None):
        return self.training_losses(x0, t=t, noise=noise).mean()

    def forward(self, x0):
        return self.training_loss(x0)

class TimeEmbedding(nn.Module):
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

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

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
    def __init__(self, channels):
        super().__init__()
        self.norm = group_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        h_norm = self.norm(x)
        q = self.q(h_norm).reshape(b, c, h * w)
        k = self.k(h_norm).reshape(b, c, h * w)
        v = self.v(h_norm).reshape(b, c, h * w)
        attn = torch.softmax(torch.einsum("bcn,bcm->bnm", q * (c ** -0.5), k), dim=-1)
        out = torch.einsum("bnm,bcm->bcn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)

class ImageBackbone(nn.Module):
    def __init__(self, image_size=32, channels=3, base_channels=128, out_channels=None,
                 channel_mult=(1, 2, 4, 8), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.0):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        time_dim = base_channels * 4
        self.init_conv = nn.Conv2d(channels, base_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(TimeEmbedding(base_channels),
                                      nn.Linear(base_channels, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))

        self.downs = nn.ModuleList()
        hs_channels = [base_channels]
        current_channels = base_channels
        current_res = image_size
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(current_channels, out_ch, time_dim, dropout))
                current_channels = out_ch
                attns.append(AttentionBlock(current_channels)
                             if current_res in attn_resolutions else nn.Identity())
                hs_channels.append(current_channels)
            down = Downsample(current_channels) if level != len(channel_mult) - 1 else nn.Identity()
            if level != len(channel_mult) - 1:
                hs_channels.append(current_channels)
                current_res //= 2
            self.downs.append(nn.ModuleList([blocks, attns, down]))

        self.mid1 = ResnetBlock(current_channels, current_channels, time_dim, dropout)
        self.mid_attn = AttentionBlock(current_channels)
        self.mid2 = ResnetBlock(current_channels, current_channels, time_dim, dropout)

        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = hs_channels.pop()
                blocks.append(ResnetBlock(current_channels + skip_ch, out_ch, time_dim, dropout))
                current_channels = out_ch
                attns.append(AttentionBlock(current_channels)
                             if current_res in attn_resolutions else nn.Identity())
            up = Upsample(current_channels) if level != 0 else nn.Identity()
            if level != 0:
                current_res *= 2
            self.ups.append(nn.ModuleList([blocks, attns, up]))

        self.final_norm = group_norm(current_channels)
        self.final_conv = zero_module(nn.Conv2d(current_channels, self.out_channels, 3, padding=1))

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

def training_step(diffusion, x0, opt, ema=None):
    loss = diffusion.training_loss(x0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if ema is not None:
        ema.update(diffusion)
    return loss.detach()

if __name__ == "__main__":
    backbone = ImageBackbone(image_size=32, base_channels=16, channel_mult=(1, 2),
                             num_res_blocks=1, attn_resolutions=(16,))
    diffusion = ImageLatentGenerator(backbone, image_size=32, latent_steps=8,
                                     prediction_type="eps", variance_type="fixedlarge",
                                     loss_type="mse")
    opt = Adam(diffusion.parameters(), lr=2e-4)
    ema = ModelEMA(diffusion)
    x0 = torch.rand(2, 3, 32, 32) * 2 - 1
    loss = training_step(diffusion, x0, opt, ema)
    samples = ema.ema_model.sample(batch_size=2)
    print(float(loss), samples.shape)
```
