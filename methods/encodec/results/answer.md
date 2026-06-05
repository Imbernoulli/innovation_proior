# EnCodec, distilled

EnCodec is an end-to-end neural audio codec: a streaming convolutional encoder–decoder with a residual-vector-quantized latent, trained with a reconstruction loss plus a multi-scale STFT discriminator, a gradient balancer that makes the loss weights interpretable, and an optional Transformer entropy model over the codes. A single model spans 1.5–24 kbps and runs faster than real time on one CPU core.

## The problem

Compress speech and music to a small discrete bitstream and reconstruct them with minimal perceptual distortion, end to end, streaming, across a range of bitrates and audio types — beating hand-engineered classical codecs (Opus/EVS).

## Key ideas

**Convolutional encoder–decoder.** Encoder: initial kernel-7 conv, then `B=4` blocks of (residual unit + strided downsampling conv with kernel = 2×stride), channels doubling per block, then a 2-layer LSTM, then a kernel-7 conv to the latent dimension. Strides (2, 4, 5, 8) → 320× downsampling → 75 latent frames/s at 24 kHz. Decoder mirrors it with transposed convolutions and reversed strides. Streaming via all-causal padding (emit `s` output steps per transposed conv immediately, buffer the rest); weight norm replaces time-dim layer norm in the streamable model.

**Residual vector quantization (RVQ).** Quantize the latent with codebook 1, take the residual, quantize it with codebook 2, and so on for `N_q` stages (up to 32 codebooks of 1024 entries = 10 bits each; 16 at 48 kHz). The code is the sum of chosen entries. Codebooks use EMA updates (decay 0.99) and dead-entry re-seeding; trained through quantization with a straight-through estimator + commitment loss. Sampling `N_q` (a multiple of 4) per batch and **dropping trailing codebooks at inference** lets one model serve 1.5/3/6/12/24 kbps.

```
RVQ:  residual = z ; out = 0
      for c in 1..N_q:  q = quantize_c(residual)
                        residual = residual − q      # refine leftover
                        out      = out + q           # codes sum to reconstruction
commitment:  l_w = Σ_c ‖ z_c − q_c(z_c) ‖₂²          # no grad to q_c
```

**Objective.** Reconstruction = time-domain ℓ₁ plus multi-scale mel (windows 2^i, i=5..11; ℓ₁ + α_i·ℓ₂, α_i=1). Perceptual = multi-scale STFT (MS-STFT) discriminator over complex STFT (windows [2048,1024,512,256,128]), trained with hinge loss + relative feature matching:

```
ℓ_t      = ‖x − x̂‖₁
ℓ_f      = (1/|α||s|) Σ_i ‖S_i(x) − S_i(x̂)‖₁ + α_i ‖S_i(x) − S_i(x̂)‖₂
ℓ_g(x̂)  = (1/K) Σ_k max(0, 1 − D_k(x̂))                                  # generator
L_d      = (1/K) Σ_k max(0, 1 − D_k(x)) + max(0, 1 + D_k(x̂))            # discriminator (hinge)
ℓ_feat   = (1/KL) Σ_k Σ_l ‖D_k^l(x) − D_k^l(x̂)‖₁ / mean(‖D_k^l(x)‖₁)     # relative feature matching
L_G      = λ_t ℓ_t + λ_f ℓ_f + λ_g ℓ_g + λ_feat ℓ_feat + λ_w l_w
```

Weights: λ_t=0.1, λ_f=1, λ_g=3, λ_feat=3 (24 kHz; λ_g=λ_feat=4 at 48 kHz), λ_w=1. Discriminator updated with probability 2/3 (24 kHz) to avoid overpowering the decoder.

**Loss balancer.** For each loss that depends on the model only via the output `x̂`, with `g_i = ∂ℓ_i/∂x̂` and `⟨‖g_i‖₂⟩_β` its EMA norm (β=0.999, R=1):

```
g̃_i = R · (λ_i / Σ_j λ_j) · g_i / ⟨‖g_i‖₂⟩_β        # backprop Σ_i g̃_i
```

This renormalizes every loss to a common gradient scale (so the discriminator's swings can't destabilize training) and makes each λ_i interpretable as the fraction of the model gradient from loss i (when Σλ=1). The commitment loss l_w is excluded (not a function of x̂).

**Entropy coding.** A small Transformer (5 layers, 8 heads, 200 channels, FFN 800) predicts each timestep's `N_q` codebook distributions in parallel from the previous step (one embedding table and one linear head per codebook), feeding a deterministic range coder (probabilities rounded to 1e-6, range width 2^24) for up to ~40% further lossless compression while staying faster than real time.

**Training:** Adam, batch 64 × 1-second clips, lr 3×10⁻⁴, β₁=0.5, β₂=0.9, 300 epochs × 2,000 updates. ELU activations.

## Working code

```python
import torch, torch.nn as nn, torch.nn.functional as F

class VectorQuantization(nn.Module):
    def __init__(self, dim, codebook_size=1024):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))   # EMA-updated in practice
    def forward(self, x):                                                # x: [B, T, D]
        d = x.pow(2).sum(-1, keepdim=True) - 2 * x @ self.codebook.t() + self.codebook.pow(2).sum(-1)
        idx = d.argmin(-1)
        q = F.embedding(idx, self.codebook)
        loss = F.mse_loss(q.detach(), x)                                 # commitment loss
        q = x + (q - x).detach()                                         # straight-through
        return q, idx, loss

class ResidualVectorQuantization(nn.Module):
    def __init__(self, dim, n_q=32, codebook_size=1024):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(dim, codebook_size) for _ in range(n_q)])
    def forward(self, x, n_q=None):
        out, residual, idxs, losses = 0.0, x, [], []
        for layer in self.layers[:(n_q or len(self.layers))]:
            q, idx, loss = layer(residual)
            residual = residual - q
            out = out + q
            idxs.append(idx); losses.append(loss)
        return out, torch.stack(idxs), torch.stack(losses)

class ResidualUnit(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(), nn.Conv1d(dim, dim // 2, 3, padding=dilation, dilation=dilation),
            nn.ELU(), nn.Conv1d(dim // 2, dim, 1))
    def forward(self, x): return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, n_filters=32, strides=(2, 4, 5, 8), dimension=128):
        super().__init__()
        layers, mult = [nn.Conv1d(1, n_filters, 7, padding=3)], 1
        for s in strides:
            layers += [ResidualUnit(mult * n_filters), nn.ELU(),
                       nn.Conv1d(mult * n_filters, mult * 2 * n_filters, 2 * s, s, padding=s // 2)]
            mult *= 2
        layers += [nn.ELU(), nn.Conv1d(mult * n_filters, dimension, 7, padding=3)]
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)
# Decoder: mirror with ConvTranspose1d, reversed strides, halving channels; LSTM over latents.

class Balancer:
    def __init__(self, weights, R=1.0, beta=0.999):
        self.weights, self.R, self.beta, self.ema = weights, R, beta, {}
    def backward(self, losses, x_hat):
        grads, norms = {}, {}
        for k, loss in losses.items():
            g, = torch.autograd.grad(loss, x_hat, retain_graph=True)
            grads[k] = g
            n = g.norm(p=2).item()
            self.ema[k] = self.beta * self.ema.get(k, n) + (1 - self.beta) * n
            norms[k] = self.ema[k]
        tw = sum(self.weights[k] for k in losses)
        out = sum(self.R * self.weights[k] / tw * grads[k] / (norms[k] + 1e-12) for k in losses)
        x_hat.backward(out)
```
