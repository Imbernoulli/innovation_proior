I want high-resolution, photorealistic images of complex natural scenes, and I want them with the properties that make likelihood-based generators worth using — broad coverage of the data distribution, stable optimization, no mode collapse — but without the brutal compute bill those models currently demand. The strongest image generators right now are denoising diffusion models: they beat GANs on sample quality, they cover the distribution instead of collapsing onto a few modes, and they are flexible, since an unconditional model can be inpainted, colorized, or guided at test time. I do not want to abandon that family; I want to make it cheap. The trouble is structural. A diffusion model defines a fixed forward process that slowly turns an image into Gaussian noise and learns to reverse it one denoising step at a time, so sampling is a long sequential walk — tens to a thousand steps — and *every* step is a full forward pass of a large UNet over the entire pixel grid. Training is worse, with a forward and backward pass at full resolution summed over the whole noise schedule. The best pixel-space models cost hundreds of GPU-days to train and days just to draw fifty thousand samples. Faster samplers chip at the step count, but every remaining step is still a full pixel-space pass and training cost is untouched.

The right question is what all that compute is being spent on. Likelihood-based models are mode-covering: they must place mass on essentially the whole dataset, so they spend capacity in proportion to the actual information content of the signal — and a natural image, bit for bit, is dominated by high-frequency texture (grass, skin pores, fabric weave) that is almost entirely imperceptible. Imagine the rate/distortion curve of an already-trained diffusion model: as it spends more capacity, distortion drops, but in two regimes. A first regime buys a large perceptual improvement that is all high-frequency detail, while the semantics — what is in the picture, how it is laid out — barely move; this is pure perceptual compression, throwing away detail no human notices. Only a second regime, with the remaining bits, does the actual semantic modeling. The expensive sequential network pays full pixel-space price during a first phase whose entire job is to discard detail that does not matter. The existing two-stage compress-then-model line (VQ-VAE through VQGAN and the large text-to-image autoregressive models) sees the same decomposition but fails on its second stage: an autoregressive transformer is quadratic in sequence length and generates tokens one at a time, so to stay tractable it forces *extreme* compression (around 16×), which makes the autoencoder discard real detail and puts a low ceiling on quality, needs billions of parameters, and flattens the 2D latent into a 1D raster order that throws away spatial structure. Jointly learning the latent and a score prior end to end avoids the transformer but introduces a delicate reconstruction-versus-prior weighting and makes the latent a moving target while the prior chases it.

I propose the Latent Diffusion Model. The leverage is that per-step cost is set by how many spatial positions the expensive model must process — roughly quadratic in resolution for convolutions, worse for attention — and the first, near-semantics-free regime eats most of the budget, so I simply take that regime away from the diffusion model. I hand the perceptual compression to a cheap, one-time autoencoder and run the entire (slow, sequential) diffusion process in a small latent space where every position already carries semantic content. Concretely, an encoder $E$ downsamples an image $x\in\mathbb R^{H\times W\times 3}$ by a factor $f$ to a latent $z\in\mathbb R^{h\times w\times c}$ with $h=H/f,\,w=W/f$, and a decoder $D$ reconstructs; this pair is trained once and frozen. One UNet pass costs work proportional to the number of positions, so going to $hw=HW/f^2$ positions cuts convolutional work per layer by $f^2$ and the dominant attention term by $f^4$, multiplied across all $T$ steps and the whole training run — a 16× saving at $f=4$, 64× at $f=8$. The crucial move that distinguishes this from the old two-stage work is the second stage: I model the latent prior with a *convolutional* diffusion model instead of an autoregressive transformer. A diffusion UNet scales gently in spatial size, not quadratically in a flattened sequence, and natively respects 2D layout, so I am no longer forced into extreme compression — I can choose a *mild* $f=4$ or $8$ where the autoencoder reconstructs almost perfectly. And I fully decouple the two stages — autoencoder first, frozen, then diffusion in a fixed latent — which removes the reconstruction-versus-prior weighting entirely, gives faithful reconstructions, and lets one universal autoencoder be reused across many diffusion trainings and tasks.

The first stage is a perceptual autoencoder. I do not train it on pixel $L_2$ or $L_1$ alone, because an $L_2$ reconstruction loss is minimized by the conditional mean of each pixel, which under uncertainty about fine detail is a blurred average — the exact opposite of the perceptual equivalence I need. Instead I combine a perceptual loss (LPIPS — comparing deep features of $x$ and its reconstruction rather than raw pixels, rewarding what a vision network finds salient) with a patch-based adversarial loss (a PatchGAN discriminator that forces reconstructions to look like real images locally, keeping them on the natural-image manifold). The reconstruction and adversarial gradients can differ wildly in magnitude and drift during training, so I weight the adversarial term *adaptively* by the ratio of the gradient norms at the last decoder layer, $d_\text{weight}=\lVert\nabla L_\text{rec}\rVert/(\lVert\nabla L_\text{adv}\rVert+\epsilon)$, clamped to a sane range, which keeps the two forces comparable with no hand-tuned schedule. The latent itself must be kept well-behaved — zero-centered, modest variance — or it becomes a nightmare to model, but only *lightly*, since heavy regularization trades away the reconstruction fidelity that is the hard ceiling on everything. Two variants do this: a KL variant where $E$ emits a diagonal Gaussian $(\mu,\sigma)$, $z=\mu+\sigma\cdot\epsilon$ with $\epsilon\sim\mathcal N(0,1)$, penalized by a tiny KL toward $\mathcal N(0,1)$ with the closed form $\tfrac12\sum(\mu^2+\sigma^2-1-\log\sigma^2)$ at weight $\sim10^{-6}$; or a VQ variant with a high-capacity codebook where $z$ is taken *before* quantization and the quantizer is folded into the decoder's first layer. I can afford to barely regularize precisely because the prior is a powerful diffusion model rather than a weak prior or sequence model that needs a tidy discrete latent.

The second stage is ordinary DDPM machinery, now on $z$. The forward process is fixed: with signal and noise schedules and $\mathrm{SNR}(t)=\alpha_t^2/\sigma_t^2$, $$q(z_t\mid z_0)=\mathcal N(z_t\mid \alpha_t z_0,\ \sigma_t^2 I),\qquad z_t=\alpha_t z_0+\sigma_t\epsilon.$$ The ELBO decomposes over $t$ into KL terms between the true Gaussian posterior $q(z_{t-1}\mid z_t,z_0)$ and the learned reverse step. I parameterize the reverse step as the true posterior with the unknown $z_0$ replaced by a network estimate $z_\theta(z_t,t)$; since both posteriors are Gaussians with the same variance, each KL collapses to a scaled squared error between means, and after substituting the posterior mean the sum becomes $$\sum_t \mathbb E_\epsilon\,\tfrac12\big(\mathrm{SNR}(t-1)-\mathrm{SNR}(t)\big)\,\lVert z_0-z_\theta(\alpha_t z_0+\sigma_t\epsilon,\,t)\rVert^2,$$ a denoising regression at each step weighted by the drop in SNR. Reparameterizing to predict the injected noise, $\epsilon_\theta=(z_t-\alpha_t z_\theta)/\sigma_t$, converts the target by $\lVert z_0-z_\theta\rVert^2=(\sigma_t^2/\alpha_t^2)\lVert\epsilon-\epsilon_\theta\rVert^2$; noise prediction is the more stable target and ties $\epsilon_\theta$, up to scale, to the score of the noised data. The principled weight is messy and $t$-dependent, and it happens to emphasize the high-SNR, trivially-denoisable steps — exactly the imperceptible-detail steps. Setting every step's weight equal de-emphasizes those and focuses capacity on the harder mid-SNR steps where semantic structure is decided, which is the rate/distortion observation baked directly into the loss. That gives the simple objective $$L_\text{LDM}=\mathbb E_{E(x),\,\epsilon\sim\mathcal N(0,1),\,t}\;\lVert\epsilon-\epsilon_\theta(z_t,t)\rVert^2,$$ with $t$ uniform. Because the forward process is closed-form, training needs no chain: encode once to $z_0=E(x)$, draw a single $z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon$, and regress. The backbone $\epsilon_\theta$ is a time-conditional 2D-convolutional UNet with self-attention only at coarse levels, exploiting the spatial structure the old AR models flattened away, and sampling decodes once at the end through $D$ — not per step — since the sequential cost lives in latent space. One scale subtlety matters: the noise schedule's SNR assumes roughly unit-variance data, but a KL latent's variance is arbitrary, so I rescale $z\leftarrow z/\hat\sigma$ by one empirical standard deviation (a single `scale_factor`, inverted at decode), keeping the schedule calibrated even when sampling convolutionally beyond the training resolution; the VQ latent already sits near unit variance.

For conditional generation $p(z\mid y)$ the cases split. When $y$ is spatially aligned with the image — a low-resolution input, a segmentation map, an inpainting mask — I downsample it to the latent grid and concatenate it to the UNet input along the channel axis, an almost-free injection where the conditioning encoder is the identity. When $y$ has no spatial correspondence at all — a text prompt, a class label, a layout — I cannot concatenate, so I let each image location pull information from anywhere in a conditioning representation via cross-attention. A domain-specific encoder $\tau_\theta$ maps $y$ to a sequence $\tau_\theta(y)\in\mathbb R^{M\times d_\tau}$ (an unmasked transformer for text, a single learnable embedding for a class label, discretized box tuples for layout), and at several UNet levels the flattened feature map is the query side while $\tau_\theta(y)$ is the key/value side: $$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big)V,\quad Q=W_Q\,\varphi_i(z_t),\ K=W_K\,\tau_\theta(y),\ V=W_V\,\tau_\theta(y).$$ The $1/\sqrt d$ factor is load-bearing: the dot products grow with the projection dimension $d$, and without it the softmax saturates so that one entry dominates and gradients through the rest vanish; dividing by $\sqrt d$ keeps the logits $O(1)$. The mechanism is blind to modality, so one machinery serves text, class, and layout, and $\tau_\theta$ and $\epsilon_\theta$ train jointly under the frozen autoencoder against $L=\mathbb E\,\lVert\epsilon-\epsilon_\theta(z_t,t,\tau_\theta(y))\rVert^2$. Finally, diffusion's test-time steering carries over and is now cheap in latent space: classifier-free guidance $\hat\epsilon=\epsilon_\theta(z,c)+s\,(\epsilon_\theta(z,c)-\epsilon_\theta(z,\varnothing))$ trades diversity for fidelity, and post-hoc guidance $\hat\epsilon\leftarrow\epsilon_\theta(z_t,t)+\sqrt{1-\alpha_t^2}\,\nabla_{z_t}\log p_\Phi(y\mid T(D(z_0(z_t))))$ with an $L_2$ or LPIPS guider $T$ gives super-resolution-style control.

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
