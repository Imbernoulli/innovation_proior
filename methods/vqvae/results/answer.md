# VQ-VAE: Neural Discrete Representation Learning

## Problem

Learn a *discrete* latent representation that keeps the high-level, semantically important structure of data (objects, phonemes, scene layout), trains with low-variance gradients comparable to the continuous reparameterization trick, stays in use even with a powerful autoregressive decoder (no posterior collapse), and reaches likelihoods competitive with continuous-latent VAEs.

## Key idea

Make the posterior a **deterministic one-hot** over a finite codebook and fix the prior **uniform**. The encoder emits a continuous vector `z_e(x)`; quantize it to the nearest codebook vector `z_q(x)=e_k`; decode from `z_q(x)`.

- With a deterministic one-hot posterior and uniform prior, `KL(q‖p)=log K` is **constant**, so it drops from the objective. This both removes the discrete-KL gradient problem and removes the mechanism behind posterior collapse — the latent channel can no longer be switched off for free, so a powerful decoder is forced to use it.
- The nearest-neighbor `argmin` is non-differentiable, so the encoder is trained with a **straight-through estimator**: forward uses `e_k`, backward copies the decoder-input gradient straight onto `z_e`. Low-variance (biased) — the property score-function (NVIL/VIMCO) and Gumbel-softmax routes never delivered cleanly.

## Final objective

For one latent (averaged over the `N` latents of a feature map for the codebook and commitment terms):

```
L = log p(x | z_q(x))  +  ‖ sg[z_e(x)] − e ‖₂²  +  β ‖ z_e(x) − sg[e] ‖₂²
```

where `z_q(x) = e_k`, `k = argmin_j ‖z_e(x) − e_j‖₂`, `e ∈ R^{K×D}`, and `sg[·]` is stop-gradient (identity forward, zero gradient backward).

- **Term 1 (reconstruction):** trains the decoder directly and the encoder via straight-through; the embeddings receive nothing here.
- **Term 2 (codebook / VQ):** online k-means — moves each code `e` toward its assigned encoder outputs (`sg` freezes the encoder).
- **Term 3 (commitment):** makes the encoder commit to its chosen code so the unbounded latent scale doesn't run away (`sg` freezes the codebook). `β=0.25` (robust over `[0.1, 2.0]`).

Straight-through is implemented as `z_q = z_e + sg[e_k − z_e]` (equals `e_k` forward; passes gradient through `z_e` backward).

## EMA codebook update (alternative to Term 2)

The codebook term is k-means, whose optimum for code `e_i` is the mean of its assigned encoder outputs. Compute it online with an EMA (optimizer-independent, usually faster), with Laplace-smoothed counts so unused codes don't die:

```
N_i ← γ N_i + (1−γ) n_i,   m_i ← γ m_i + (1−γ) Σ_j z_{i,j},   e_i = m_i / N_i,   γ=0.99
```

With EMA, only the commitment term (Term 3) remains in the loss.

## Generation

The prior is held uniform during training. Afterward, fit an expressive **autoregressive prior** over the learned discrete codes — a PixelCNN over the 2D latent grid for images (spatial masking only, single channel), a WaveNet over the 1D code sequence for audio — then ancestral-sample codes and decode.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Codebook with the three-term loss and straight-through gradient."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.D, self.K, self.beta = embedding_dim, num_embeddings, commitment_cost
        self.embedding = nn.Embedding(self.K, self.D)              # e in R^{K x D}
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

    def forward(self, z_e):                       # z_e: [B, D, H, W]
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e.view(-1, self.D)

        # ||z_e - e||^2 = ||z_e||^2 - 2 z_e.e + ||e||^2
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(1))
        idx = dist.argmin(1)                                       # nearest code
        z_q = self.embedding(idx).view(z_e.shape)                 # z_q = e_k

        codebook_loss = F.mse_loss(z_q, z_e.detach())            # term 2: e -> z_e
        commit_loss   = F.mse_loss(z_q.detach(), z_e)            # term 3: z_e -> e
        vq_loss = codebook_loss + self.beta * commit_loss

        z_q = z_e + (z_q - z_e).detach()                          # straight-through
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, vq_loss, idx


class VectorQuantizerEMA(nn.Module):
    """Codebook updated by EMA k-means; only the commitment term is in the loss."""
    def __init__(self, num_embeddings, embedding_dim,
                 commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.D, self.K = embedding_dim, num_embeddings
        self.beta, self.decay, self.eps = commitment_cost, decay, eps
        embed = torch.randn(self.K, self.D)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(self.K))  # N_i
        self.register_buffer("ema_w", embed.clone())               # m_i

    def forward(self, z_e):
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e.view(-1, self.D)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embedding.t()
                + self.embedding.pow(2).sum(1))
        idx = dist.argmin(1)
        onehot = F.one_hot(idx, self.K).type(flat.dtype)
        z_q = self.embedding[idx].view(z_e.shape)

        if self.training:
            n = onehot.sum(0)
            self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
            dw = onehot.t() @ flat
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            N = self.cluster_size.sum()
            cluster = (self.cluster_size + self.eps) / (N + self.K * self.eps) * N
            self.embedding.copy_(self.ema_w / cluster.unsqueeze(1))

        commit_loss = self.beta * F.mse_loss(z_q.detach(), z_e)
        z_q = z_e + (z_q - z_e).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, commit_loss, idx


class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq):
        super().__init__()
        self.encoder, self.decoder, self.vq = encoder, decoder, vq

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, idx = self.vq(z_e)
        x_rec = self.decoder(z_q)
        return x_rec, vq_loss, idx


def train_step(model, x, optimizer):
    optimizer.zero_grad()
    x_rec, vq_loss, _ = model(x)
    loss = F.mse_loss(x_rec, x) + vq_loss     # reconstruction + (codebook + commit)
    loss.backward()
    optimizer.step()
    return loss
```

The encoder/decoder are standard strided-conv + residual-block stacks; after training, a PixelCNN/WaveNet is fit over the index field `idx` to sample new codes for generation.
