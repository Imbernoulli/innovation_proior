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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)


def extract(a, t, x_shape):
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


class Diffusion(nn.Module):
    def __init__(self, model, image_size, channels, timesteps=1000):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps

        betas = linear_beta_schedule(timesteps).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x0, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def p_sample(self, x_t, t_int):
        t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
        eps = self.model(x_t, t)
        alpha_t = extract(self.alphas, t, x_t.shape)
        beta_t = extract(self.betas, t, x_t.shape)
        alpha_cumprod_t = extract(self.alphas_cumprod, t, x_t.shape)
        mean = (x_t - beta_t / torch.sqrt(1.0 - alpha_cumprod_t) * eps) / torch.sqrt(alpha_t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        return mean + nonzero_mask * torch.sqrt(extract(self.posterior_variance, t, x_t.shape)) * noise

    @torch.no_grad()
    def sample(self, batch_size, device):
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        for t_int in reversed(range(self.timesteps)):
            x = self.p_sample(x, t_int)
        return x

    def loss(self, x0):
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        return F.mse_loss(self.model(x_t, t), noise)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000.0) / (half - 1)))
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def group_norm(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.res(x)


class UNet(nn.Module):
    def __init__(self, image_size=32, channels=3, base=128, mult=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        self.channels = channels
        time_dim = base * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base),
            nn.Linear(base, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim))
        self.init_conv = nn.Conv2d(channels, base, 3, padding=1)

        self.downs = nn.ModuleList()
        chs = [base]
        cur = base
        for i, m in enumerate(mult):
            out = base * m
            for _ in range(2):
                self.downs.append(ResBlock(cur, out, time_dim, dropout))
                cur = out
                chs.append(cur)
            if i != len(mult) - 1:
                self.downs.append(nn.Conv2d(cur, cur, 3, stride=2, padding=1))

        self.mid = ResBlock(cur, cur, time_dim, dropout)

        self.ups = nn.ModuleList()
        for i, m in reversed(list(enumerate(mult))):
            out = base * m
            for _ in range(2):
                self.ups.append(ResBlock(cur + chs.pop(), out, time_dim, dropout))
                cur = out
            if i != 0:
                next_out = base * mult[i - 1]
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(cur, next_out, 3, padding=1)))
                cur = next_out

        self.out = nn.Conv2d(cur, channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.init_conv(x)
        skips = [h]
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
                skips.append(h)
            else:
                h = layer(h)
        h = self.mid(h, t_emb)
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = torch.cat((h, skips.pop()), dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)
        return self.out(h)


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for name, value in self.ema_model.state_dict().items():
            src = model.state_dict()[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                value.copy_(src)


if __name__ == "__main__":
    unet = UNet(image_size=32, base=32, mult=(1, 2), dropout=0.0)
    diffusion = Diffusion(unet, image_size=32, channels=3, timesteps=8)
    opt = Adam(diffusion.parameters(), lr=2e-4)
    ema = ModelEMA(diffusion)
    x0 = torch.rand(2, 3, 32, 32) * 2.0 - 1.0
    loss = diffusion.loss(x0)
    loss.backward()
    opt.step()
    ema.update(diffusion)
    samples = ema.ema_model.sample(batch_size=2, device=x0.device)
    print(float(loss), samples.shape)
```
