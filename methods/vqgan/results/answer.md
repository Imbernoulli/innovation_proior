# VQGAN

## Problem

Transformers model long-range structure by attending over all pairs of inputs, but that costs `O(n²)` in sequence length. For images the sequence is the pixels and its length grows quadratically with resolution, so a pixel-space transformer is infeasible past tiny images. CNNs scale (local kernels, linear cost) but their locality bias makes them weak at the global composition that high-resolution synthesis needs.

## Key idea

A two-stage model that splits the labor. **Stage 1 (VQGAN)** is a convolutional vector-quantized autoencoder that compresses an image into a short grid of discrete codes, trained so the reconstruction is *perceptually faithful even at high compression*. **Stage 2** is an autoregressive GPT-style transformer that models the prior over the grid of code indices. The CNN owns local perceptual detail and compression; the transformer owns global composition over a sequence short enough for full attention. Sampling = transformer draws code indices, frozen decoder renders pixels; a sliding attention window scales it to megapixels.

## Stage 1 — VQGAN

Encoder `E`, decoder `G`, codebook `Z = {z_k}_{k=1}^{K} ⊂ ℝ^{n_z}`. Encode `ẑ = E(x) ∈ ℝ^{h×w×n_z}`, quantize each spatial vector to its nearest code, decode:

- `z_q = q(ẑ) = (argmin_{z_k∈Z} ‖ẑ_{ij} − z_k‖)`, `x̂ = G(z_q)`.
- Straight-through gradient: `z_q = ẑ + sg[z_q − ẑ]` (forward `z_q`, backward identity onto `E(x)`).
- VQ loss: `L_VQ = L_rec + ‖sg[E(x)] − z_q‖² + ‖z_q − sg[E(x)]‖²` (codebook + commitment; commitment weight = 1.0).

The change from a plain VQVAE is the reconstruction objective. Replace pixel `L₂` (which blurs under heavy compression) with a **perceptual loss** `L_rec = LPIPS(x, x̂)` (+ a small pixel `L₁`), and add a **patch-based adversarial loss** with a discriminator `D` (PatchGAN, judging `N×N` patches):

`Q* = argmin_{E,G,Z} max_D  E_{x∼p(x)} [ L_VQ + λ·L_GAN ]`, with `L_GAN = log D(x) + log(1−D(x̂))` (trained with a hinge surrogate in practice).

**Adaptive GAN weight** — the load-bearing equation. Balance the perceptual and adversarial gradients at the decoder's last layer `G_L`:

`λ = ‖∇_{G_L}[L_rec]‖ / (‖∇_{G_L}[L_GAN]‖ + δ)`, `δ = 10⁻⁴`,

clamped to `[0, 10⁴]` and detached. This normalizes the GAN gradient to the magnitude of the reconstruction gradient at the point where they meet, so neither dominates regardless of their raw scales or how they drift during training. `λ = 0` during a warm-up (≥ 1 epoch) so adversarial pressure only starts once the autoencoder reconstructs sanely.

Architecture: ResNet-style `E`/`G` (GroupNorm + Swish, `m` downsampling/upsampling stages so `f = 2^m`, `h = H/2^m`), one non-local self-attention block at the lowest resolution, `1×1` convs to/from `n_z`, PatchGAN discriminator. Typical: `K = 1024`, `f = 16`, latent `16×16`.

## Stage 2 — transformer prior over codes

With `E,G,Z` frozen, an image is a sequence of indices `s ∈ {0,…,K−1}^{h×w}` in **raster (row-major) order**. A GPT-2-style decoder-only transformer models `p(s) = Πᵢ p(sᵢ | s_{<i})`; train by minimizing `L_Transformer = E[−log p(s)]` (cross-entropy next-index prediction). **Conditioning**: `p(s|c) = Πᵢ p(sᵢ | s_{<i}, c)`. A class label is a single prepended token; a spatial condition (segmentation/depth/edges/low-res image) is encoded by its own VQGAN into indices `r` and prepended, with the loss restricted to the `s` part — one decoder-only mechanism for every task. **Sampling**: autoregressively sample indices (temperature `t = 1.0`, top-`k ≈ 100`), map to codebook vectors, decode with `G`. **Megapixels**: train on latent crops and sample with a **sliding attention window** over the raster grid (condition on coordinates when the data is aligned and unconditional).

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e, self.e_dim, self.beta = n_e, e_dim, beta
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0/n_e, 1.0/n_e)

    def forward(self, z):                                # z: (B, C, h, w)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.e_dim)
        d = (z_flat**2).sum(1, keepdim=True) + (self.embedding.weight**2).sum(1) \
            - 2 * z_flat @ self.embedding.weight.t()
        idx = torch.argmin(d, dim=1)
        z_q = self.embedding(idx).view(z.shape)
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        z_q = z + (z_q - z).detach()                     # straight-through
        return z_q.permute(0, 3, 1, 2).contiguous(), loss, idx


def hinge_d_loss(real, fake):
    return 0.5 * (F.relu(1. - real).mean() + F.relu(1. + fake).mean())


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, perceptual_weight=1.0, disc_weight=1.0):
        super().__init__()
        self.perceptual_loss = LPIPS().eval()            # VGG-feature perceptual distance
        self.discriminator = PatchDiscriminator()        # PatchGAN
        self.codebook_weight = codebook_weight
        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight
        self.disc_start = disc_start

    def calculate_adaptive_weight(self, rec_loss, g_loss, last_layer):
        rec_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        g_grad   = torch.autograd.grad(g_loss,   last_layer, retain_graph=True)[0]
        w = torch.norm(rec_grad) / (torch.norm(g_grad) + 1e-4)
        return (self.disc_weight * torch.clamp(w, 0.0, 1e4)).detach()

    def forward(self, codebook_loss, x, x_hat, optimizer_idx, global_step, last_layer):
        rec = (torch.abs(x - x_hat) + self.perceptual_weight * self.perceptual_loss(x, x_hat)).mean()
        disc_on = 1.0 if global_step >= self.disc_start else 0.0
        if optimizer_idx == 0:                           # generator (autoencoder)
            g_loss = -self.discriminator(x_hat).mean()
            lam = self.calculate_adaptive_weight(rec, g_loss, last_layer) if disc_on else 0.0
            return rec + lam * disc_on * g_loss + self.codebook_weight * codebook_loss.mean()
        if optimizer_idx == 1:                           # discriminator
            return disc_on * hinge_d_loss(self.discriminator(x.detach()),
                                          self.discriminator(x_hat.detach()))


class VQModel(nn.Module):
    def __init__(self, ddconfig, lossconfig, n_embed, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.loss = VQLPIPSWithDiscriminator(**lossconfig)

    def encode(self, x):
        return self.quantize(self.quant_conv(self.encoder(x)))

    def decode(self, z_q):
        return self.decoder(self.post_quant_conv(z_q))

    def forward(self, x):
        z_q, diff, _ = self.encode(x)
        return self.decode(z_q), diff

    def get_last_layer(self):
        return self.decoder.conv_out.weight              # G_L for the adaptive lambda


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=24, n_head=16, n_embd=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        t = idx.size(1)
        x = self.ln_f(self.blocks(self.tok_emb(idx) + self.pos_emb[:, :t, :]))
        logits = self.head(x)
        loss = None if targets is None else \
            F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
# Block: LayerNorm -> causal multi-head self-attention -> LayerNorm -> 4x-wide GELU MLP, both residual.


class Net2NetTransformer(nn.Module):
    def __init__(self, first_stage, transformer):
        super().__init__()
        self.first_stage = first_stage                   # frozen VQGAN
        self.transformer = transformer

    @torch.no_grad()
    def encode_to_z(self, x):
        quant, _, idx = self.first_stage.encode(x)
        return quant, idx.view(x.shape[0], -1)           # raster order

    def forward(self, x, c):                             # c: condition index prefix
        _, z_idx = self.encode_to_z(x)
        cz = torch.cat((c, z_idx), dim=1)
        logits, _ = self.transformer(cz[:, :-1])
        return logits[:, c.shape[1]-1:], z_idx

    @torch.no_grad()
    def sample(self, c, steps, temperature=1.0, top_k=100):
        x = c
        for _ in range(steps):
            x_cond = x[:, -self.transformer.block_size:]
            logits, _ = self.transformer(x_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            x = torch.cat((x, torch.multinomial(probs, 1)), dim=1)
        return x[:, c.shape[1]:]
```

Training: two Adam optimizers (`betas=(0.5, 0.9)`) alternate the autoencoder/generator update and the discriminator update for stage 1; stage 2 trains the transformer by cross-entropy on the (optionally conditioned) index sequence with the VQGAN frozen.
