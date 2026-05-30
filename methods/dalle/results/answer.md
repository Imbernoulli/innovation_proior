# DALL-E

## Problem

High-fidelity, flexible text-to-image generation controllable by free-form natural language, from a single general model trained at scale — rather than task-specific architectures tuned on small datasets.

## Key idea

Model text and image as one autoregressive token stream with a single transformer. Because pixels make the image sequence far too long (and likelihood objectives over pixels waste capacity on short-range detail), first compress each image into a short grid of discrete tokens, then model the tokens. Two stages, which are coordinate ascent on one ELBO over images `x`, captions `y`, and image tokens `z`, factorized `p_{θ,ψ}(x, y, z) = p_θ(x | y, z) p_ψ(y, z)`:

`ln p_{θ,ψ}(x, y) ≥ E_{z∼q_φ(z|x)}[ ln p_θ(x | y, z) − β·D_KL(q_φ(y, z | x) ‖ p_ψ(y, z)) ]`.

- **Stage 1 — discrete VAE (dVAE).** A convolutional ResNet encoder compresses a 256×256×3 image into a 32×32 grid of tokens, each from a vocabulary of `K = 8192` (a 192× shorter sequence; large vocabulary limits information loss). The encoder's categorical output is non-differentiable, so it is relaxed with **Gumbel-Softmax**: add Gumbel noise to the logits and softmax at temperature `τ`, annealed `1 → 1/16` so it hardens to the true categorical. The pixel likelihood is the **logit-Laplace** distribution (bounded support matching pixel range). Trained on images alone (optimize `φ, θ`).
- **Stage 2 — transformer prior.** Freeze the dVAE. BPE-encode the caption (≤256 tokens, vocab 16384), extract the 1024 image tokens by argmax of the encoder logits, concatenate into one stream, and train a 12-billion-parameter **decoder-only sparse transformer** (row/column/convolutional attention masks; image tokens carry row+column embeddings) autoregressively (optimize `ψ`).

Generate by sampling token streams and reranking the decoded images with a pretrained contrastive image-text model (e.g. N = 512 candidates).

## Key formulas

- Gumbel-Softmax code: `soft = softmax((logits + g)/τ)`, `g ∼ Gumbel(0,1)`, `z = soft · codebook`; tight as `τ → 0`.
- Logit-Laplace pdf on (0,1): `f(x | μ, b) = 1/(2 b x (1−x)) · exp(−|logit(x) − μ| / b)`; the decoder emits 6 maps (3 `μ`, 3 `ln b`); pixels pre-mapped by `φ: x ↦ ((1−2ε)/255)x + ε`, `ε = 0.1`; reconstruct `x̂ = φ⁻¹(sigmoid(μ))`.
- `β` ramped 0→6.6 (improves codebook usage and reconstruction here); effective per-position KL weight `β/192`.
- Stage-2 loss: count-normalized cross-entropy, weighted `1/8` text + `7/8` image.

## Architecture / hyperparameters

dVAE: conv ResNet, bottleneck resblocks (mostly 3×3, 1×1 on channel-changing skips), 7×7 input conv, 1×1 output conv to 8192 logits, max-pool downsampling (×3 → /8), nearest-neighbor upsampling decoder, 1×1 convs around the bottleneck, `n_hid = 256`. AdamW (β₁=0.9, β₂=0.999, ε=1e-8, wd=1e-4), EMA 0.999, τ annealed over 150k updates, step size 1e-4→1.25e-6, 3M updates, batch 512. Transformer: 64 layers, 62 heads, head size 64 (`d_model = 3968`); convolutional attention (11×11) only in the last layer, else column attention when `(i−2) mod 4 = 0` else row; learned per-position padding tokens. AdamW (β₁=0.9, β₂=0.96, wd=4.5e-2), grad-clip norm 4, step size 4.5e-4, batch 1024, 430k updates.

## Code

```python
import torch
from torch import nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, n_in, n_out, n_layers):
        super().__init__()
        n_hid = n_out // 4
        self.post_gain = 1 / (n_layers ** 2)
        self.id_path = nn.Conv2d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.res_path = nn.Sequential(
            nn.ReLU(), nn.Conv2d(n_in,  n_hid, 3, padding=1),
            nn.ReLU(), nn.Conv2d(n_hid, n_hid, 3, padding=1),
            nn.ReLU(), nn.Conv2d(n_hid, n_hid, 3, padding=1),
            nn.ReLU(), nn.Conv2d(n_hid, n_out, 1))
    def forward(self, x):
        return self.id_path(x) + self.post_gain * self.res_path(x)


class Encoder(nn.Module):
    def __init__(self, n_hid=256, n_blk=2, in_ch=3, vocab_size=8192):
        super().__init__()
        n_layers = 4 * n_blk
        blk = lambda i, o: EncoderBlock(i, o, n_layers)
        self.blocks = nn.Sequential(
            nn.Conv2d(in_ch, n_hid, 7, padding=3),
            *[blk(n_hid, n_hid) for _ in range(n_blk)], nn.MaxPool2d(2),
            blk(n_hid, 2*n_hid), *[blk(2*n_hid, 2*n_hid) for _ in range(n_blk-1)], nn.MaxPool2d(2),
            blk(2*n_hid, 4*n_hid), *[blk(4*n_hid, 4*n_hid) for _ in range(n_blk-1)], nn.MaxPool2d(2),
            blk(4*n_hid, 8*n_hid), *[blk(8*n_hid, 8*n_hid) for _ in range(n_blk-1)],
            nn.ReLU(), nn.Conv2d(8*n_hid, vocab_size, 1))
    def forward(self, x):
        return self.blocks(x)            # [B, 8192, 32, 32]


def gumbel_softmax_codes(logits, codebook, tau):
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
    soft = F.softmax((logits + g) / tau, dim=1)
    z = torch.einsum("bkhw,kd->bdhw", soft, codebook)
    return soft, z


def logit_laplace_nll(x, mu, ln_b):
    b = ln_b.exp()
    return (torch.log(2 * b * x * (1 - x)) + (torch.logit(x) - mu).abs() / b).mean()


def kl_uniform(soft):
    q = soft.mean(dim=(0, 2, 3))
    K = soft.shape[1]
    return (q * (q.add(1e-9).log() + torch.log(torch.tensor(float(K))))).sum()


def phi(img, eps=0.1):                   # map [0,1] pixels into (eps, 1-eps)
    return (1 - 2 * eps) * img + eps


# Stage 1: train dVAE on images alone (relaxed ELBO)
opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()),
                        lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
for step, img in enumerate(loader):
    x = phi(img)
    logits = encoder(x)
    tau  = anneal(step, 1.0, 1/16, 150_000)
    beta = anneal(step, 0.0, 6.6, 5_000)
    soft, z = gumbel_softmax_codes(logits, codebook, tau)
    mu, ln_b = decoder(z).chunk(2, dim=1)         # 6 maps: 3 mu, 3 ln b
    loss = logit_laplace_nll(x, mu, ln_b) + beta * kl_uniform(soft)
    opt.zero_grad(); loss.backward(); opt.step()


# Stage 2: fit transformer prior over concatenated text+image tokens
@torch.no_grad()
def image_tokens(encoder, img):
    return encoder(phi(img)).argmax(dim=1).flatten(1)     # [B, 1024], argmax (no gumbel)

opt = torch.optim.AdamW(transformer.parameters(), lr=4.5e-4, betas=(0.9, 0.96),
                        eps=1e-8, weight_decay=4.5e-2)
for text_tok, img in loader:                              # text_tok: [B, 256]
    img_tok = image_tokens(encoder, img)                  # [B, 1024]
    stream = torch.cat([text_tok, img_tok + text_vocab], dim=1)
    logits = transformer(stream[:, :-1])
    tgt = stream[:, 1:]
    n_text = 256 - 1
    ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
    loss = (1/8) * ce[:, :n_text].mean() + (7/8) * ce[:, n_text:].mean()
    torch.nn.utils.clip_grad_norm_(transformer.parameters(), 4.0)
    opt.zero_grad(); loss.backward(); opt.step()
```
