# Latent Diffusion Models (LDM)

## Problem

Denoising diffusion models give the best image sample quality and full
distribution coverage, but they run in pixel space: every one of the many
denoising steps is a full UNet evaluation over the full-resolution RGB grid, in
both directions during training. Training costs hundreds of GPU-days and
sampling is slow. Most of that compute is wasted: a likelihood-based model spends
capacity in proportion to information content, and natural images are dominated
by imperceptible high-frequency detail, so the model burns its budget in a
"perceptual compression" regime that removes detail without touching semantics.

## Key idea

Split the two jobs. Train a one-time autoencoder that does the perceptual
compression — mapping an image to a lower-dimensional, perceptually equivalent
latent and back — then freeze it and run an ordinary diffusion model entirely in
that latent space. Per-step cost drops by roughly the square of the spatial
downsampling factor f, while the diffusion model now operates where every
position already carries semantics. Because the latent prior is modeled by a
*convolutional* diffusion model (not an autoregressive transformer, which would
be quadratic in sequence length and force extreme compression), the autoencoder
needs only *mild* compression (f = 4–8), keeping reconstructions faithful and
preserving the latent's 2D structure.

## Method

**Stage 1 — perceptual autoencoder, trained once then frozen.** Encoder E
downsamples x ∈ R^{H×W×3} by f to z ∈ R^{h×w×c}; decoder D reconstructs.
Objective: pixel L1 + LPIPS perceptual loss + a patch-based (PatchGAN)
adversarial loss, with the adversarial term weighted adaptively by the ratio of
reconstruction- to adversarial-gradient norms at the last decoder layer. Pure
pixel losses, especially L2, reward averaging when fine detail is uncertain; the
perceptual + adversarial terms keep reconstructions sharp and on the image
manifold. The latent is
*lightly* regularized in one of two ways:
- **KL**: E emits a diagonal Gaussian (μ, σ); z = μ + σ·ε; a tiny KL penalty
  (weight ~1e-6) toward N(0,1), with closed form ½Σ(μ² + σ² − 1 − log σ²).
- **VQ**: a vector-quantization layer with a large codebook; z is taken before
  quantization and the quantizer is absorbed into the decoder (a VQGAN whose
  quantizer is D's first layer).
Regularization is kept minimal so reconstruction fidelity — the ceiling on final
quality — is preserved.

**Stage 2 — diffusion in latent space.** Standard DDPM machinery on z. Forward:
z_t = α_t z_0 + σ_t ε, implemented with α_t = √ᾱ_t and σ_t = √(1−ᾱ_t). The ELBO
decomposes over t into weighted denoising regressions
½(SNR(t−1)−SNR(t))‖z_0 − z_θ‖²; reparameterizing to noise prediction
(ε_θ = (z_t − α_t z_θ)/σ_t) converts the squared error by
‖z_0 − z_θ‖² = (σ_t²/α_t²)‖ε − ε_θ‖². Reweighting every step equally gives the
simple objective

  L_LDM = E_{E(x), ε∼N(0,1), t} ‖ ε − ε_θ(z_t, t) ‖².

The equal-weight, noise-prediction form de-emphasizes the trivially-denoisable
high-SNR steps and focuses capacity on the perceptually relevant ones. The
backbone ε_θ is a time-conditional 2D-convolutional UNet (self-attention at
coarse levels only), which uses the latent's spatial structure. The latent is
rescaled to unit variance (z ← z/σ̂, a scalar scale_factor) so the noise
schedule's SNR is calibrated; sampling decodes once at the end with D.

**Conditioning.** For p(z|y):
- spatially aligned y (low-res image, segmentation, mask): downsample to the
  latent grid and **concatenate** to the UNet input (τ = identity).
- arbitrary modalities (text, class, layout): a domain-specific encoder τ_θ maps
  y to a sequence τ_θ(y) ∈ R^{M×d_τ}, injected via **cross-attention** at
  several UNet levels, with image features as queries and τ_θ(y) as keys/values:

  Attention(Q,K,V) = softmax(QKᵀ/√d)·V,
  Q = W_Q φ_i(z_t), K = W_K τ_θ(y), V = W_V τ_θ(y).

  The 1/√d factor keeps the dot-product logits O(1) as the projection
  dimension grows, preventing softmax saturation.
  Conditional objective: L = E_{E(x),y,ε,t} ‖ ε − ε_θ(z_t, t, τ_θ(y)) ‖²,
  with τ_θ and ε_θ trained jointly (autoencoder frozen).

**Guidance** carries over from diffusion: classifier-free guidance
ε̂ = ε_θ(z,c) + s·(ε_θ(z,c) − ε_θ(z,∅)) for fidelity; and post-hoc image
guiding ε̂ ← ε_θ(z_t,t) + √(1−α_t²) ∇_{z_t} log p_Φ(y|T(D(z_0(z_t)))) with an
L2 or LPIPS guider T for super-resolution-style control — all cheap in latent
space.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DiagonalGaussian:
    def __init__(self, parameters):
        self.mean, logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.var = torch.exp(self.logvar)
        self.std = torch.exp(0.5 * self.logvar)
    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)
    def mode(self):
        return self.mean
    def kl(self):
        return 0.5 * torch.sum(self.mean**2 + self.var - 1.0 - self.logvar,
                               dim=[1, 2, 3])


class AutoencoderKL(nn.Module):
    """Perceptual compressor: downsample by f to a low-dim latent, and back."""
    def __init__(self, encoder, decoder, z_channels, embed_dim):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
    def encode(self, x):
        return DiagonalGaussian(self.quant_conv(self.encoder(x)))
    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))
    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior


class LPIPSWithDiscriminator(nn.Module):
    """Stage-1 loss: L1 + LPIPS + adaptively-weighted PatchGAN + tiny KL."""
    def __init__(self, perceptual, discriminator, kl_weight=1e-6, perceptual_weight=1.0):
        super().__init__()
        self.perceptual, self.discriminator = perceptual, discriminator
        self.kl_weight, self.perceptual_weight = kl_weight, perceptual_weight
        self.logvar = nn.Parameter(torch.zeros(()))
    def adaptive_weight(self, rec_loss, g_loss, last_layer):
        rg = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        gg = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        return torch.clamp(torch.norm(rg) / (torch.norm(gg) + 1e-4), 0.0, 1e4).detach()
    def forward(self, x, x_rec, posterior, last_layer):
        rec = torch.abs(x - x_rec)
        p_loss = self.perceptual(x, x_rec)
        while p_loss.ndim < rec.ndim:
            p_loss = p_loss[..., None]
        rec = rec + self.perceptual_weight * p_loss
        nll = (rec / torch.exp(self.logvar) + self.logvar).mean()
        g_loss = -self.discriminator(x_rec).mean()
        d_w = self.adaptive_weight(nll, g_loss, last_layer)
        return nll + d_w * g_loss + self.kl_weight * posterior.kl().mean()


class CrossAttention(nn.Module):
    """softmax(QK^T/sqrt(d)) V; Q from image feats, K/V from tau(y)."""
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner, context_dim = dim_head * heads, context_dim or query_dim
        self.heads, self.scale = heads, dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, query_dim)
    def forward(self, x, context=None):
        context = context if context is not None else x
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads),
                      (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        out = torch.einsum('b i j, b j d -> b i d', sim.softmax(dim=-1), v)
        return self.to_out(rearrange(out, '(b h) n d -> b n (h d)', h=self.heads))


class LatentDiffusion(nn.Module):
    def __init__(self, autoencoder, unet, cond_encoder=None,
                 timesteps=1000, scale_factor=1.0, conditioning='crossattn'):
        super().__init__()
        self.first_stage = autoencoder.eval()
        for p in self.first_stage.parameters():
            p.requires_grad_(False)
        self.unet, self.cond_encoder = unet, cond_encoder
        self.conditioning, self.scale_factor = conditioning, scale_factor
        self.num_timesteps = timesteps
        betas = torch.linspace(1e-4, 2e-2, timesteps)
        acp = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer('sqrt_acp', torch.sqrt(acp))
        self.register_buffer('sqrt_one_minus_acp', torch.sqrt(1.0 - acp))

    def _extract(self, buffer, t, x):
        return buffer[t].view(-1, *([1] * (x.ndim - 1)))

    @torch.no_grad()
    def encode_to_latent(self, x):
        return self.scale_factor * self.first_stage.encode(x).sample()

    @torch.no_grad()
    def decode_latent(self, z):
        return self.first_stage.decode(z / self.scale_factor)

    def q_sample(self, z0, t, noise):
        a = self._extract(self.sqrt_acp, t, z0)
        b = self._extract(self.sqrt_one_minus_acp, t, z0)
        return a * z0 + b * noise

    def get_conditioning(self, y):
        if self.cond_encoder is None or y is None:
            return y
        return self.cond_encoder(y)

    def apply_model(self, z_t, t, cond):
        if self.conditioning == 'concat' and cond is not None:
            return self.unet(torch.cat([z_t, cond], dim=1), t)
        return self.unet(z_t, t, context=cond)

    def forward(self, x, y=None):
        z0 = self.encode_to_latent(x)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        cond = self.get_conditioning(y)
        return F.mse_loss(self.apply_model(z_t, t, cond), noise)

    @torch.no_grad()
    def sample(self, shape, y=None, device=None):
        device = device or self.sqrt_acp.device
        z = torch.randn(tuple(shape), device=device)
        cond = self.get_conditioning(y)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((z.shape[0],), i, device=device, dtype=torch.long)
            eps = self.apply_model(z, t, cond)
            a = self._extract(self.sqrt_acp, t, z)
            b = self._extract(self.sqrt_one_minus_acp, t, z)
            z0 = (z - b * eps) / a                    # predict z_0 from eps
            if i > 0:
                t_prev = torch.full((z.shape[0],), i - 1, device=device, dtype=torch.long)
                ap = self._extract(self.sqrt_acp, t_prev, z)
                bp = self._extract(self.sqrt_one_minus_acp, t_prev, z)
                z = ap * z0 + bp * eps                # DDIM-style step
            else:
                z = z0
        return self.decode_latent(z)
```
